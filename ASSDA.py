import os
import sys
import random
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

from source.dataset import hierarchical_dataset, get_dataloader
from source.model import Model

from modules.domain_adapt import d_cls_inst
from modules.radam import AdamW, RAdam

from utils.converter import AttnLabelConverter, CTCLabelConverter
from utils.averager import Averager

from test import validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_local_features(opt,
                          source_context_history, source_prediction,
                          target_context_history, target_prediction):
    feature_dim = source_context_history.size()[-1]

    source_feature = source_context_history.reshape(-1, feature_dim)
    target_feature = target_context_history.reshape(-1, feature_dim)

    # print(type(pred_class),pred_class)
    source_pred_score, source_pred_class = source_prediction.max(-1)
    target_pred_score, target_pred_class = target_prediction.max(-1)
    source_valid_char_index = (source_pred_score.reshape(-1, ) > opt.pc).nonzero().reshape(-1, )
    source_valid_char_feature = source_feature.reshape(-1, feature_dim).index_select(0, source_valid_char_index)
    target_valid_char_index = (target_pred_score.reshape(-1, ) > opt.pc).nonzero().reshape(-1, )
    target_valid_char_feature = target_feature.reshape(-1, feature_dim).index_select(0, target_valid_char_index)

    return source_valid_char_feature, target_valid_char_feature


def setup_optimizer(opt, filtered_parameters, global_discriminator_params, local_discriminator_params):
    # setup optimizer
    if opt.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(filtered_parameters, lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
        d_image_opt = optim.SGD(global_discriminator_params, lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
        d_inst_opt = optim.SGD(local_discriminator_params,
                                lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    elif opt.optimizer.lower() == 'adam':
        optimizer = AdamW(filtered_parameters, lr=opt.lr, betas=(opt.beta1, opt.beta2),
                                weight_decay=opt.weight_decay)
        d_image_opt = AdamW(global_discriminator_params, lr=opt.lr,
                                betas=(opt.beta1, opt.beta2),
                                weight_decay=opt.weight_decay)
        d_inst_opt = AdamW(local_discriminator_params,
                                betas=(opt.beta1, opt.beta2),
                                weight_decay=opt.weight_decay)

    elif opt.optimizer.lower() == 'radam':
        optimizer = RAdam(filtered_parameters, lr=opt.lr,
                                betas=(opt.beta1, opt.beta2),
                                weight_decay=opt.weight_decay)
        d_image_opt = RAdam(global_discriminator_params, lr=opt.lr,
                                betas=(opt.beta1, opt.beta2),
                                weight_decay=opt.weight_decay)
        d_inst_opt = RAdam(local_discriminator_params,
                                betas=(opt.beta1, opt.beta2),
                                weight_decay=opt.weight_decay)

    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=0.1 * opt.lr, rho=opt.rho,
                                            eps=opt.eps)
        d_image_opt = optim.Adadelta(global_discriminator_params,
                                            lr=opt.lr,
                                            rho=opt.rho,
                                            eps=opt.eps)
        d_inst_opt = optim.Adadelta(local_discriminator_params,
                                            lr=opt.lr,
                                            rho=opt.rho,
                                            eps=opt.eps)

    return optimizer, d_image_opt, d_inst_opt


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

    source_loader = get_dataloader(opt, source_data, opt.batch_size, shuffle = True)
    target_loader = get_dataloader(opt, target_data, opt.batch_size, shuffle = True, mode = "raw")
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

    # initialize domain classifiers here.
    global_discriminator = d_cls_inst(fc_size=13312)
    local_discriminator = d_cls_inst(fc_size=256)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    global_discriminator = torch.nn.DataParallel(global_discriminator).to(device)
    local_discriminator = torch.nn.DataParallel(local_discriminator).to(device)
    model.train()
    global_discriminator.train()
    local_discriminator.train()

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
    D_criterion = torch.nn.BCEWithLogitsLoss().to(device)

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
    optimizer, d_image_opt, d_inst_opt  = setup_optimizer(opt, filtered_parameters, 
                                                            global_discriminator.parameters(), 
                                                            local_discriminator.parameters())

    print("Start Adapting...\n")
    main_log += "Start Adapting...\n"

    # set up iter dataloader
    source_loader_iter = iter(source_loader)
    target_loader_iter = iter(target_loader)

    # loss averager
    cls_loss_avg = Averager()
    sim_loss_avg = Averager()
    loss_avg = Averager()

    # training loop
    gamma = 0
    omega = 1
    start_iter = 0
    best_score = float('-inf')
    score_descent = 0

    for iteration in tqdm(
            range(start_iter, opt.total_iter + 1),
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
            global_discriminator.train()
            local_discriminator.train()

            if (current_score >= best_score):
                score_descent = 0

                best_score = current_score
                torch.save(model.state_dict(), f"./trained_model/{opt.approach}/{opt.model}_best_ASSDA.pth")
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
            sim_loss_avg.reset()
        
        if iteration == opt.total_iter:
            main_log += f'Stop training at iteration: {iteration}!\n'
            print(f'Stop training at iteration: {iteration}!\n')
            break

        if opt.decay_flag and iteration > (opt.total_iter // 2):
            d_image_opt.param_groups[0]['lr'] -= (opt.lr / (opt.total_iter // 2))
            d_inst_opt.param_groups[0]['lr'] -= (opt.lr / (opt.total_iter // 2))

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
        src_preds, src_global_feature, src_local_feature = model(images_source_tensor, labels_source_index[:, :-1])

        # src_global_feature = model.visual_feature
        # src_local_feature = model.Prediction.context_history
        target = labels_source_index[:, 1:]  # without [GO] Symbol
        src_cls_loss = criterion(src_preds.view(-1, src_preds.shape[-1]), target.contiguous().view(-1))
        src_global_feature = src_global_feature.reshape(src_global_feature.shape[0], -1)
        src_local_feature = src_local_feature.view(-1, src_local_feature.shape[-1])

        text_for_pred = (
                    torch.LongTensor(len(images_target_tensor))
                    .fill_(opt.sos_token_index)
                    .to(device)
                )
        tar_preds, tar_global_feature, tar_local_feature = model(images_target_tensor, text_for_pred, is_train=False)

        # tar_global_feature = model.visual_feature
        # tar_local_feature = model.Prediction.context_history
        tar_global_feature = tar_global_feature.reshape(tar_global_feature.shape[0], -1)
        tar_local_feature = tar_local_feature.view(-1, tar_local_feature.shape[-1])

        src_local_feature, tar_local_feature = filter_local_features(opt, src_local_feature, src_preds, 
                                                                        tar_local_feature, tar_preds)

        # add domain adaption elements
        # setup hyperparameter
        if iteration % 2000 == 0:
            p = float(iteration + start_iter) / opt.total_iter
            gamma = 2. / (1. + np.exp(-10 * p)) - 1
            omega = 1 - 1. / (1. + np.exp(-10 * p))
        global_discriminator.module.set_beta(gamma)
        local_discriminator.module.set_beta(gamma)

        src_d_img_score = global_discriminator(src_global_feature)
        src_d_inst_score = local_discriminator(src_local_feature)
        tar_d_img_score = global_discriminator(tar_global_feature)
        tar_d_inst_score = local_discriminator(tar_local_feature)

        src_d_img_loss = D_criterion(src_d_img_score, torch.zeros_like(src_d_img_score).to(device))
        src_d_inst_loss = D_criterion(src_d_inst_score, torch.zeros_like(src_d_inst_score).to(device))
        tar_d_img_loss = D_criterion(tar_d_img_score, torch.ones_like(tar_d_img_score).to(device))
        tar_d_inst_loss = D_criterion(tar_d_inst_score, torch.ones_like(tar_d_inst_score).to(device))
        d_img_loss = src_d_img_loss + tar_d_img_loss
        d_inst_loss = src_d_inst_loss + tar_d_inst_loss

        # add domain loss
        loss = src_cls_loss.mean() + omega * (d_img_loss.mean() + d_inst_loss.mean())
        loss_avg.add(loss)
        cls_loss_avg.add(src_cls_loss)
        sim_loss_avg.add(d_img_loss + d_inst_loss)

        # set gradient to zero...
        model.zero_grad(set_to_none=True)
        # Domain classifiers
        global_discriminator.zero_grad(set_to_none=True)
        local_discriminator.zero_grad(set_to_none=True)
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        # frcnn optimizer update
        optimizer.step()
        # domain optimizer update
        d_inst_opt.step()
        d_image_opt.step()


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
    # # Optimization options
    parser.add_argument('--adam', action='store_true',
                        help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--optimizer', type=str, default='adadelta',
                        help='optimizer type: adam , Radam, Adadelta')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate, default=0.1 for adam')
    parser.add_argument('--decay_flag', action='store_true', help='for learning rate decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.9')
    # parser.add_argument('--weight_decay', type=float, default=0.9, help='weight_decay for adam. default=0.9')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--pc', type=float, default=0.0,
                        help='confidence threshold,, 0,0.1,0.2,0.4,0.8.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--rho', type=float, default=0.95,
                        help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
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