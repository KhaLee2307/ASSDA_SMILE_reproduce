import os
import sys
import time
import argparse
import re
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nltk.metrics.distance import edit_distance

from utils.averager import Averager
from utils.converter import AttnLabelConverter, CTCLabelConverter

from source.dataset import hierarchical_dataset, AlignCollate
from source.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark_all_eval(model, criterion, converter, opt):
    """evaluation with 6 benchmark evaluation datasets"""
    eval_data_list = [
        "IIIT5k_3000",
        "SVT",
        "IC13_1015",
        "IC15_2077",
        "SVTP",
        "CUTE80",
    ]
    if (opt.addition == True):
        eval_data_list = [
            "COCOv1.4",
            "Uber",
            "ArT",
            "ReCTS",
        ]
    if (opt.exception == True):
        eval_data_list = [
            "IIIT5k_3000",
            "SVT",
            "IC13_857",
            "IC15_1811",
            "SVTP",
            "CUTE80",
        ]
    if (opt.all == True):
        eval_data_list = [
            "IIIT5k_3000",
            "SVT",
            "IC13_857",
            "IC13_1015",
            "IC15_1811",
            "IC15_2077",
            "SVTP",
            "CUTE80",
        ]
    if (opt.union == True):
        eval_data_list = [
            "artistic",
            "contextless",
            "curve",
            "general",
            # "incomplete",
            # "incomplete_ori",
            # "multi_oriented",
            "multi_words",
            "salient",
        ]

    accuracy_list = []
    total_forward_time = 0
    total_eval_data_number = 0
    total_correct_number = 0
    dashed_line = "-" * 80
    print(dashed_line)
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_eval = AlignCollate(opt)
        eval_data, eval_data_log = hierarchical_dataset(
            root=eval_data_path, opt=opt
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_eval,
            pin_memory=True,
        )

        _, accuracy_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, eval_loader, converter, opt, tqdm_position=0
        )
        accuracy_list.append(f"{accuracy_by_best_model:0.2f}")
        total_forward_time += infer_time
        total_eval_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        print(f"Acc {accuracy_by_best_model:0.2f}")
        print(dashed_line)

    averaged_forward_time = total_forward_time / total_eval_data_number * 1000
    total_accuracy = total_correct_number / total_eval_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    eval_log = "accuracy: "
    for name, accuracy in zip(eval_data_list, accuracy_list):
        eval_log += f"{name}: {accuracy}\t"
    eval_log += f"total_accuracy: {total_accuracy:0.2f}\t"
    eval_log += f"averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.2f}"
    print(eval_log)

    # for convenience
    print("\t".join(accuracy_list))
    print(f"Total_accuracy:{total_accuracy:0.2f}")

    return total_accuracy, eval_data_list, accuracy_list


def validation(model, criterion, eval_loader, converter, opt, tqdm_position=1):
    """validation or evaluation"""
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    # for i, (image_tensors, labels) in tqdm(
    #     enumerate(eval_loader),
    #     total=len(eval_loader),
    #     position=tqdm_position,
    #     leave=False,
    # ):
    for image_tensors, labels in eval_loader:
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # for max length prediction
        labels_index, labels_length = converter.encode(
            labels, batch_max_length=opt.batch_max_length
        )
        
        if "CTC" in opt.Prediction:
            start_time = time.time()
            preds, _, _ = model(image)
            forward_time = time.time() - start_time

            # calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = criterion(
                preds.log_softmax(2).permute(1, 0, 2),
                labels_index,
                preds_size,
                labels_length,
            )
        else:
            text_for_pred = (
                torch.LongTensor(batch_size).fill_(converter.dict["[SOS]"]).to(device)
            )

            start_time = time.time()
            preds, _, _ = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            target = labels_index[:, 1:]  # without [SOS] Symbol
            cost = criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1),
            )

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_size = torch.IntTensor([preds.size(1)] * preds_index.size(0)).to(device)
        preds_str = converter.decode(preds_index, preds_size)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, prd, prd_max_prob in zip(labels, preds_str, preds_max_prob):
            if "Attn" in opt.Prediction:
                prd_EOS = prd.find("[EOS]")
                prd = prd[:prd_EOS]  # prune after "end of sentence" token ([EOS])
                prd_max_prob = prd_max_prob[:prd_EOS]

            """
            In our experiment, if the model predicts at least one [UNK] token, we count the word prediction as incorrect.
            To not take account of [UNK] token, use the below line.
            prd = prd.replace('[UNK]', '') 
            """

            # to evaluate 'case sensitive model' with alphanumeric and case insensitve setting. = same with ASTER
            gt = gt.lower()
            prd = prd.lower()
            alphanumeric_case_insensitve = "0123456789abcdefghijklmnopqrstuvwxyz"
            out_of_alphanumeric_case_insensitve = f"[^{alphanumeric_case_insensitve}]"
            gt = re.sub(out_of_alphanumeric_case_insensitve, "", gt)
            prd = re.sub(out_of_alphanumeric_case_insensitve, "", prd)

            if opt.NED:
                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(prd) == 0:
                    norm_ED += 0
                elif len(gt) > len(prd):
                    norm_ED += 1 - edit_distance(prd, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(prd, gt) / len(prd)

            else:
                if prd == gt:
                    n_correct += 1

            # calculate confidence score (= multiply of prd_max_prob)
            try:
                confidence_score = prd_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([EOS])
            confidence_score_list.append(confidence_score)

    if opt.NED:
        # ICDAR2019 Normalized Edit Distance. In web page, they report % of norm_ED (= norm_ED * 100).
        score = norm_ED / float(length_of_data) * 100
    else:
        score = n_correct / float(length_of_data) * 100  # accuracy

    return (
        valid_loss_avg.val(),
        score,
        preds_str,
        confidence_score_list,
        labels,
        infer_time,
        length_of_data,
    )


def test(opt):
    """model configuration"""
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
        opt.sos_token_index = converter.dict["[SOS]"]
        opt.eos_token_index = converter.dict["[EOS]"]
    opt.num_class = len(converter.character)

    model = Model(opt)
    print(
        "model input parameters",
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction,
    )
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print("loading pretrained model from %s" % opt.saved_model)
    try:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    except:
        print(
            "*** pretrained model not match strictly *** and thus load_state_dict with strict=False mode"
        )
        # pretrained_state_dict = torch.load(opt.saved_model)
        # for name in pretrained_state_dict:
        #     print(name)
        model.load_state_dict(
            torch.load(opt.saved_model, map_location=device), strict=False
        )

    """ setup loss """
    if "CTC" in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        # ignore [PAD] token
        criterion = torch.nn.CrossEntropyLoss(ignore_index=converter.dict["[PAD]"]).to(
            device
        )

    """ evaluation """
    model.eval()
    with torch.no_grad():
        # evaluate 6 benchmark evaluation datasets
        benchmark_all_eval(model, criterion, converter, opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_data", 
        default="data/test/benchmark/",
        help="path to evaluation dataset",
    )
    parser.add_argument("--addition", action='store_true', default=False, help='test on addition data')
    parser.add_argument("--exception", action='store_true', default=False, help='test on exception data')
    parser.add_argument("--all", action='store_true', default=False, help='test on all data')
    parser.add_argument("--union", action='store_true', default=False, help='test on Union14 data')
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=512, help="input batch size")
    parser.add_argument(
        "--saved_model", required=True, help="path to saved_model to evaluation"
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

    test(opt)