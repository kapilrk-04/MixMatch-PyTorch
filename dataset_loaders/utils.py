import numpy as np
from PIL import Image

import torchvision
import torch

'''
Generate k augmentations for each unlebeled image.
Change this class to output 3 images for K=3 agumentations
and output 4 images for K=4 augmentations
'''
class TransformKAugs(object):
    def __init__(self, transform, K):
        self.transform = transform
        self.K = K

    def __call__(self, inp):
        out = []
        for i in range(self.K):
            out.append(self.transform(inp))
        return out


mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)

def normalize(x, mean=mean, std=std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

# Crop image randomly
class RandomPadandCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

# Flip image randomly
class RandomFlip(object):
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

# Add gaussian noise to the image
class GaussianNoise(object):
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x