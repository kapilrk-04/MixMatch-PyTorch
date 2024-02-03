'''
CIFAR100 dataset classes
'''

import numpy as np
from PIL import Image

import torchvision
import torch

import dataset_loaders.utils as utils

cifar100_mean = (0.4914, 0.4822, 0.4465)
cifar100_std = (0.2471, 0.2435, 0.2616)

class CIFAR100Labeled(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100Labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = utils.transpose(utils.normalize(self.data, cifar10_mean, cifar10_std))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100Unlabeled(CIFAR100Labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100Unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
