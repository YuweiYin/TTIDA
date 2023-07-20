# -*- coding:utf-8 -*-
"""
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
"""

import random
from PIL import ImageFilter
import torchvision.transforms as transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MoCoAugTransforms:
    """
    # MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
    # https://arxiv.org/pdf/1911.05722.pdf
    # https://github.com/facebookresearch/moco
    """

    def __init__(self, mean, std):
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )
        normalize = transforms.Normalize(
            mean=mean, std=std
        )

        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        self.augmentation_v1 = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        self.augmentation_v2 = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
