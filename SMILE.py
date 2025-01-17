import os
import sys
import random
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

from source.dataset import hierarchical_dataset, get_dataloader
from source.model import Model

from utils.converter import AttnLabelConverter, CTCLabelConverter
from utils.averager import Averager

from test import validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch_entropy_loss(p_softmax):
    entropy = - torch.mul(p_softmax, torch.log(p_softmax))
    entropy =  torch.sum(entropy, dim=1)
    return entropy


def main(opt):
    dashed_line = "-" * 80
    main_log = ""
    opt_log = dashed_line + "\n"

    """ create folder for log and trained model """
    if (not os.path.exists(f'log/{opt.approach}/')):
        os.makedirs(f'log/{opt.approach}/')
    if (not os.path.exists(f'trained_model/{opt.approach}/')):
        os.makedirs(f'trained_model/{opt.approach}/')

    # source data
    source_data, source_data_log = hierarchical_dataset(opt.source_data, opt)
    opt_log += source_data_log

    # target data
    target_data_list = list(np.load(opt.select_data))
    target_data, target_data_log = hierarchical_dataset(opt.target_data, opt, mode = "raw")
    target_data = Subset(target_data, target_data_list)
    opt_log += target_data_log

    # valid data
    valid_data, valid_data_log = hierarchical_dataset(opt.valid_data, opt)
    opt_log += valid_data_log

    source_loader = get_dataloader(opt, source_data, opt.batch_size, shuffle = True, mode = "label", aug = True)
    target_loader = get_dataloader(opt, target_data, opt.batch_size, shuffle = True, mode = "raw", aug = opt.aug)
    valid_loader = get_dataloader(opt, valid_data, opt.batch_size_val, shuffle = False)

    del source_data, source_data_log, target_data, target_data_log, valid_data, valid_data_log
    del target_data_list

    """ model configuration """
    if opt.Prediction == "CTC":
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
        opt.sos_token_index = converter.dict["[SOS]"]
        opt.eos_token_index = converter.dict["[EOS]"]
    opt.num_class = len(converter.character)

    # setup model
    model = Model(opt)
    opt_log += "Init model\n"

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    # load pretrained model
    pretrained = torch.load(opt.saved_model)
    model.load_state_dict(pretrained)
    opt_log += "Load pretrained model\n"

    del pretrained

    """ setup loss """
    if opt.Prediction == "CTC":
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        # ignore [PAD] token
        criterion = torch.nn.CrossEntropyLoss(ignore_index=converter.dict["[PAD]"]).to(device)

    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f"Trainable params num: {sum(params_num)}")
    opt_log += f"Trainable params num: {sum(params_num)}"

    del params_num

    """ final options """
    opt_log += "------------ Options -------------\n"
    args = vars(opt)
    for k, v in args.items():
        if str(k) == "character" and len(str(v)) > 500:
            opt_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
        else:
            opt_log += f"{str(k)}: {str(v)}\n"
    opt_log += "---------------------------------------\n"
    print(opt_log)
    main_log += opt_log

    # set up optimizer
    optimizer = torch.optim.AdamW(filtered_parameters, lr=opt.lr, weight_decay = 0.01)

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt.lr,
                cycle_momentum=False,
                div_factor=20,
                final_div_factor=1000,
                total_steps=opt.total_iter,
            )
    
    print("Start Adapting...\n")
    main_log += "Start Adapting...\n"

    # set up iter dataloader
    source_loader_iter = iter(source_loader)
    target_loader_iter = iter(target_loader)

    # loss averager
    cls_loss_avg = Averager()
    em_loss_avg = Averager()
    loss_avg = Averager()

    # training loop
    best_score = float('-inf')
    score_descent = 0

    # pace parameter
    tar_portion = opt.init_portion
    add_portion = opt.add_portion
    tar_lambda = opt.tar_lambda

    for iteration in tqdm(
        range(0, opt.total_iter + 1),
        total=opt.total_iter,
        position=0,
        leave=True,
    ):
        if (iteration % opt.val_interval == 0 or iteration == opt.total_iter):
            # valiation part
            model.eval()
            with torch.no_grad():
                (
                    valid_loss,
                    current_score,
                    preds,
                    confidence_score,
                    labels,
                    infer_time,
                    length_of_data,
                ) = validation(model, criterion, valid_loader, converter, opt)
            model.train()

            if (current_score >= best_score):
                score_descent = 0

                best_score = current_score
                torch.save(model.state_dict(), f"./trained_model/{opt.approach}/{opt.model}_best_SMILE.pth")
            else:
                score_descent += 1

            # log
            lr = optimizer.param_groups[0]["lr"]
            valid_log = f'\nValidation at {iteration}/{opt.total_iter}:\n'
            valid_log += f'Train_loss: {loss_avg.val():0.4f}, Valid_loss: {valid_loss:0.4f}, '
            valid_log += f'Current_lr: {lr:0.5f}, '
            valid_log += f'Current_score: {current_score:0.2f}, Best_score: {best_score:0.2f}, '
            valid_log += f'Score_descent: {score_descent}\n'
            print(valid_log)

            main_log += valid_log

            main_log += dashed_line

            loss_avg.reset()
            cls_loss_avg.reset()
            em_loss_avg.reset()

        if iteration == opt.total_iter:
            main_log += f'Stop training at iteration: {iteration}!\n'
            print(f'Stop training at iteration: {iteration}!\n')
            break

        """ source domain """
        try:
            images_source_tensor, labels_source = next(source_loader_iter)
        except StopIteration:
            del source_loader_iter
            source_loader_iter = iter(source_loader)
            images_source_tensor, labels_source = next(source_loader_iter)

        images_source_tensor = images_source_tensor.to(device)
        labels_source_index, labels_source_length = converter.encode(
            labels_source, batch_max_length=opt.batch_max_length
        )

        """ target domain """
        try:
            images_target_tensor = next(target_loader_iter)
        except StopIteration:
            del target_loader_iter
            target_loader_iter = iter(target_loader)
            images_target_tensor = next(target_loader_iter)

        images_target_tensor = images_target_tensor.to(device)

        # Attention # align with Attention.forward
        src_preds, src_global_feature, src_local_feature = model(images_source_tensor, labels_source_index[:, :-1]) # align with Attention.forward

        target = labels_source_index[:, 1:]  # without [SOS] Symbol
        src_cls_loss = criterion(src_preds.view(-1, src_preds.shape[-1]), target.contiguous().view(-1))

        text_for_pred = (
                    torch.LongTensor(len(images_target_tensor))
                    .fill_(opt.sos_token_index)
                    .to(device)
                )
        tar_preds, tar_global_feature, tar_local_feature = model(images_target_tensor, text_for_pred, is_train=False)

        # target entropy minimization
        tar_preds = torch.nn.functional.softmax(tar_preds.view(-1, tar_preds.shape[-1]), dim=-1)

        # self-paced procedure
        tar_em_loss = get_batch_entropy_loss(tar_preds)
        if tar_portion < 1.0 :
            tar_portion = min(tar_portion + add_portion, 1)
            tar_preds_max_prob, tar_preds_class = tar_preds.max(dim=1)

            class_set = torch.unique(tar_preds_class)[1:] # without [SOS] Symbol
            choosed_ent_pool = torch.tensor([], device=device)
            for c in class_set: 
                mask = tar_preds_class == c
                tar_ent_pool = tar_em_loss[mask]
                k = max(int(len(tar_ent_pool) * tar_portion), 1)

                choosed_ent, _ = tar_ent_pool.topk(k, largest=False)
                choosed_ent_pool = torch.cat((choosed_ent_pool, choosed_ent), 0)

            loss = src_cls_loss.mean() + choosed_ent_pool.mean() * tar_lambda
        else:
            loss = src_cls_loss.mean() + tar_em_loss.mean() * tar_lambda

        model.zero_grad(set_to_none=True)        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), opt.grad_clip
        )   # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(loss)
        cls_loss_avg.add(src_cls_loss)
        em_loss_avg.add(tar_em_loss)

        scheduler.step()

    # save log
    print(main_log, file = open(f'log/{opt.approach}/log_domain_adaptation.txt', 'w'))


if __name__ == '__main__':
    """ Argument """ 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_data",
        default="data/train/synth/",
        help="path to source dataset",
    )
    parser.add_argument(
        "--target_data",
        default="data/train/real/",
        help="path to adaptation dataset",
    )
    parser.add_argument(
        "--valid_data",
        default="data/val/",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--saved_model",
        required=True, 
        help="path to saved_model to evaluation"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--batch_size_val", type=int, default=512, help="input batch size val")
    parser.add_argument("--total_iter", type=int, default=50000, help="number of iterations to train for each round")
    parser.add_argument("--val_interval", type=int, default=500, help="Interval between each validation")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5, help="gradient clipping value. default=5"
    )
    """ Data processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        help="character label",
    )
    parser.add_argument(
        "--NED", action="store_true", help="For Normalized edit_distance"
    )
    """ Model Architecture """
    parser.add_argument("--model", type=str, required=True, help="CRNN|TRBA") 
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=3,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )
    """ Optimizer """
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate, 0.001 for Adam",
    )
    """ Experiment """
    parser.add_argument('--manual_seed', type=int, default=111, help='for random seed setting')
    """ Adaptation """
    parser.add_argument(
        "--select_data",
        required=True,
        help="path to select data",
    )
    parser.add_argument("--approach", required = True, help="select indexing approach")
    parser.add_argument("--aug", action='store_true', default=False, help='augmentation or not')
    parser.add_argument('--init_portion', type=float, default=0.5,
                        help='the size of initial target portion')
    parser.add_argument('--add_portion', type=float, default=0.0001,
                        help='the adding portion of self-paced learning')
    parser.add_argument('--tar_lambda', type=float, default=1.0,
                        help='the weight of the target domain loss')

    opt = parser.parse_args()

    opt.use_IMAGENET_norm = False  # for CRNN and TRBA
    
    if opt.model == "CRNN":  # CRNN = NVBC
        opt.Transformation = "None"
        opt.FeatureExtraction = "VGG"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "CTC"

    elif opt.model == "TRBA":  # TRBA
        opt.Transformation = "TPS"
        opt.FeatureExtraction = "ResNet"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "Attn"

    elif opt.model == "ABINet": #ABINet
        opt.use_IMAGENET_norm = True

    """ Seed and GPU setting """   
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    if sys.platform == "win32":
        opt.workers = 0

    opt.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience

    command_line_input = " ".join(sys.argv)
    print(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}"
    )

    main(opt)