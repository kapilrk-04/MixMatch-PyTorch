'''
Functions to load the dataset and split it into labelled, unlabelled and validation sets.
'''

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from dataset_loaders import cifar10, cifar100

dataset_dir = './data'

# function to call the dataset loader
def dataset_loader(dataset):
    if dataset == 'cifar10':
        dataset_dataset = cifar10.cifar10_loader()
    elif dataset == 'cifar100':
        dataset_dataset = cifar100.cifar100_loader()

    return dataset_dataset

def split_dataset(dataset, labeled_n, batch_size, k):
    labels = np.array(dataset.targets)

    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    # for each label in the dataset
    for i in range(len(np.unique(labels))):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:labeled_n//len(np.unique(labels))])
        train_unlabeled_idxs.extend(idxs[labeled_n//len(np.unique(labels)):int(-0.1*len(idxs))])
        val_idxs.extend(idxs[int(-0.1*len(idxs)):])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    train_labeled_dataset = cifar10.CIFAR10_labeled(dataset_dir, train_labeled_idxs, train=True, transform=cifar10.random_transforms(), download=False)
    train_unlabeled_dataset = cifar10.CIFAR10_unlabeled(dataset_dir, train_unlabeled_idxs, train=True, transform=cifar10.random_transforms(), download=False)
    val_dataset = cifar10.CIFAR10_labeled(dataset_dir, val_idxs, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = cifar10.CIFAR10_labeled(dataset_dir, train=False, transform=transforms.ToTensor(), download=False)

    train_labeled_loader = data.DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    train_unlabeled_loader = data.DataLoader(train_unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_labeled_loader, train_unlabeled_loader, val_loader, train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_loader, test_dataset
