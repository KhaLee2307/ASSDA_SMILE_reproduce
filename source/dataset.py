import os
import sys
import six
import PIL
import lmdb

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchvision.transforms

from .rand_aug import Augmentor

_MEAN_IMAGENET = torch.tensor([0.485, 0.456, 0.406])
_STD_IMAGENET  = torch.tensor([0.229, 0.224, 0.225])


def get_dataloader(opt, dataset, batch_size, shuffle = False, mode = "label", aug = False):
    """
    Get dataloader for each dataset

    Parameters
    ----------
    opt: argparse.ArgumentParser().parse_args()
    dataset: torch.utils.data.Dataset
    batch_size: int
    shuffle: boolean

    Returns
    ----------
    data_loader: torch.utils.data.DataLoader
    """

    if mode == "raw":
        myAlignCollate = AlignCollateRaw(opt, aug)
    else:
        myAlignCollate = AlignCollate(opt, aug)

    data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=opt.workers,
            collate_fn=myAlignCollate,
            pin_memory=False,
            drop_last=False,
        )
    return data_loader


def hierarchical_dataset(root, opt, mode="label", drop_data=[]):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f"dataset_root:    {root}\t dataset:"
    print(dataset_log)
    dataset_log += "\n"

    listdir = list()
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        if not dirnames:
            # print(dirpath)
            flag = True
            for u in drop_data:
                if u in dirpath:
                    flag = False
                    break
            if flag == True:
                listdir.append(dirpath)

    listdir.sort()

    for dirpath in listdir:
        if mode == "raw":
            # load data without label
            dataset = LmdbDataset_raw(dirpath, opt)
        else:
            # load data with label
            dataset = LmdbDataset(dirpath, opt)
        sub_dataset_log = f"sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}"
        print(sub_dataset_log)
        dataset_log += f"{sub_dataset_log}\n"
        dataset_list.append(dataset)

    # concatenate many dataset
    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class Pseudolabel_Dataset(Dataset):
    """
    Assign pseudo labels to data

    Parameters
    ----------
    unlabel_dataset: torch.utils.data.Dataset
    psudolabel_list: list(object) of pseudo labels
    """

    def __init__(self, unlabel_dataset, psudolabel_list):
        self.unlabel_dataset = unlabel_dataset
        self.psudolabel_list = psudolabel_list
        self.nSamples= len(self.psudolabel_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        label = self.psudolabel_list[index]
        img = self.unlabel_dataset[index]
        return (img, label)


class AlignCollate(object):
    """ Transform data to the same format """
    def __init__(self, opt, aug = False):
        self.opt = opt
        # resize image
        if (aug == True):
            self.transform = Rand_augment()
        else:
            self.transform = torchvision.transforms.Compose([])

        self.resize = ResizeNormalize(opt)
        print("Use Text_augment", self.transform)

    def __call__(self, batch):
        images, labels = zip(*batch)

        image_tensors = [self.resize(self.transform(image)) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


class AlignCollateRaw(object):
    """ Transform data to the same format """
    def __init__(self, opt, aug = False):
        self.opt = opt
        # resize image
        if (aug == True):
            self.transform = Rand_augment()
        else:  
            self.transform = torchvision.transforms.Compose([])

        self.resize = ResizeNormalize(opt)
        print("Use Text_augment", self.transform)

    def __call__(self, batch):
        images = batch

        image_tensors = [self.resize(self.transform(image)) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors


class AlignCollateHDGE(object):
    """ Transform data to the same format """
    def __init__(self, opt, infer=False):
        self.opt = opt

        # for transforming the input image
        if infer == False:
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize((opt.load_height,opt.load_width)),
                torchvision.transforms.RandomCrop((opt.crop_height,opt.crop_width)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize((opt.crop_height,opt.crop_width)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.transform = transform

    def __call__(self, batch):
        images = batch

        image_tensors = [self.transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors


class LmdbDataset(Dataset):
    """ Load data from Lmdb file with label """
    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = "label-%09d".encode() % index
                label = txn.get(label_key).decode("utf-8")

                # length filtering
                length_of_label = len(label)
                if length_of_label > opt.batch_max_length:
                    continue

                self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = "label-%09d".encode() % index
            label = txn.get(label_key).decode("utf-8")
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGB")

            except IOError:
                print(f"Corrupted image for {index}")
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))
                label = "[dummy_label]"

        return (img, label)


class LmdbDataset_raw(Dataset):
    """ Load data from Lmdb file without label """
    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            self.index_list = [index + 1 for index in range(self.nSamples)]

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.index_list[index]

        with self.env.begin(write=False) as txn:
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGB")

            except IOError:
                print(f"Corrupted image for {img_key}")
                # make dummy image for corrupted image.
                img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))

        return img


class ResizeNormalize(object):

    def __init__(self, opt):
        self.opt = opt
        _transforms = []

        _transforms.append(
            torchvision.transforms.Resize((self.opt.imgH, self.opt.imgW),
                               interpolation=torchvision.transforms.InterpolationMode.BICUBIC))
        _transforms.append(torchvision.transforms.ToTensor())
        if self.opt.use_IMAGENET_norm:
            _transforms.append(torchvision.transforms.Normalize(mean=_MEAN_IMAGENET,
                                                    std=_STD_IMAGENET))
        else:
            _transforms.append(torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                std=[0.5, 0.5, 0.5]))
        self._transforms = torchvision.transforms.Compose(_transforms)

    def __call__(self, image):
        image = self._transforms(image)

        return image


class Weak_augment(object):

    def __init__(self):
        augmentation = []
        augmentation.append(
            torchvision.transforms.ColorJitter(brightness=0.2,
                                   contrast=0.1,
                                   saturation=0.1,
                                   hue=0.05))
        self.Augment = torchvision.transforms.Compose(augmentation)

    def __call__(self, image):
        image = self.Augment(image)

        return image


class Rand_augment(object):

    def __init__(self):
        self.first_augmentor = Augmentor(2, 5, 'spatial')
        self.augmentor = Augmentor(2, 10, 'channel')

    def __call__(self, image):
        image = self.first_augmentor(image)
        image = self.augmentor(image)

        return image