'''
Load specific datasets using torchvision.dataset classes for the specific datasets
Specify load instructions to add new datasets to the training pipeline.
Current implementation only includes CIFAR10 and CIFAR100 datasets.
'''

import numpy as np
from PIL import Image

import torchvision
import torch

from dataset_loaders.utils import TransformKAugs
from dataset_loaders.cifar10 import CIFAR10Labeled, CIFAR10Unlabeled
from  dataset_loaders.cifar100 import CIFAR100Labeled, CIFAR100Unlabeled

def get_data(root, dataset, n_labeled, K=2,
             transform_train=None, transform_val=None,
                 download=True):

    base_dataset = None

    if dataset == 'cifar10':
        base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
        Data_labeled = CIFAR10Labeled
        Data_unlabeled = CIFAR10Unlabeled
    elif dataset == 'cifar100':
        base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)
        Data_labeled = CIFAR100Labeled
        Data_unlabeled = CIFAR100Unlabeled
    else:
        print('Dataset not found')
        exit()

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/10))

    train_labeled_dataset = Data_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = Data_unlabeled(root, train_unlabeled_idxs, train=True, transform=TransformKAugs(transform_train, K))
    val_dataset = Data_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = Data_labeled(root, train=False, transform=transform_val, download=True)

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs
