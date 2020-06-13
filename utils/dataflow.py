import os 
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.dataset import *

def get_transforms(CONFIG):
    # CIFAR transforms
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG.mean, CONFIG.std)
        ])

    val_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG.mean, CONFIG.std)
        ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CONFIG.mean, CONFIG.std)
        ])

    return train_transform, val_transform, test_transform




def get_dataset(train_transform, val_transform, test_transform, CONFIG):
    """
    TODO:
    val split
    """
    if CONFIG.dataset == "cifar10":
        train_dataset, val_dataset, test_dataset = get_cifar10(train_transform, val_transform, test_transform, CONFIG)

    elif CONFIG.dataset == "cifar100":
        train_dataset, val_dataset, test_dataset = get_cifar100(train_transform, val_transform, test_transform, CONFIG)

    elif CONFIG.dataset == "imagenet_lmdb":
        train_dataset, val_dataset, test_dataset = get_imagenet_lmdb(train_transform, val_transform, test_transform, CONFIG)

    elif CONFIG.dataset == "imagenet":
        train_dataset, val_dataset, test_dataset = get_imagenet(train_transform, val_transform, test_transform, CONFIG)

    else:
       raise

    return train_dataset, val_dataset, test_dataset


def get_dataloader(train_dataset, val_dataset, test_dataset, CONFIG):
    def _build_loader(dataset, shuffle, sampler=None):
        return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=CONFIG.batch_size,
                    pin_memory=True,
                    num_workers=CONFIG.num_workers,
                    sampler=sampler
                )
    train_sampler, val_sampler = None, None
    if CONFIG.train_portion is not None:
        train_sampler, val_sampler = split_data(train_dataset, CONFIG)

    train_loader = _build_loader(train_dataset, True, sampler=train_sampler)
    val_loader = _build_loader(train_dataset, True, sampler=val_sampler)
    test_loader = _build_loader(test_dataset, True)

    return train_loader, val_loader, test_loader


def split_data(train_dataset, CONFIG):
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(CONFIG.train_portion * num_train))

    train_idx, val_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return train_sampler, val_sampler


