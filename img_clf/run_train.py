#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import time
import logging
import argparse

import torch
from torch.backends import cudnn

from trainer_img_clf import ImageClassificationTrainer
from img_clf.utils import set_seed


def main() -> bool:
    timer_start = time.process_time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42, help="random seed for all modules {7, 17, 42}")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="cifar100", help="dataset name")
    parser.add_argument("--net", type=str, default="resnet101", help="net name")
    parser.add_argument("--epoch", type=int, default=-1, help="max epoch")
    parser.add_argument("--b", type=int, default=128, help="batch size for dataloader")
    parser.add_argument("--warm", type=int, default=10, help="warm up epochs in the training phase")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("--resume", action="store_true", default=False, help="resume training")

    parser.add_argument("--embed_size", type=int, default=300, help="dimension of word embedding vectors")
    parser.add_argument("--hidden_size", type=int, default=512, help="dimension of lstm hidden states")
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers in lstm")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="whether to use pretrained embedding for decoder")
    parser.add_argument("--stochastic", action="store_true", default=False,
                        help="stochastic or deterministic generator")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")

    parser.add_argument("--n_img_train", type=int, default=-1, help="the number of images for training. -1: all")
    parser.add_argument("--n_img_val", type=int, default=-1, help="the number of images for validation. -1: all")
    parser.add_argument("--n_img_test", type=int, default=-1, help="the number of images for testing. -1: all")

    parser.add_argument("--n_img_syn", type=int, default=0, help="the number of synthetic images per class. -1: all")
    parser.add_argument("--syn_type", type=str, default="base",
                        help="glide_base 64x64 or glide_upsample 256x256 or gan_base 64x64")

    parser.add_argument("--long_tail", action="store_true", default=False)
    parser.add_argument("--adversarial", action="store_true", default=False)
    parser.add_argument("--adv_as_test", action="store_true", default=False)  # use adv images as test set
    parser.add_argument("--extra_transforms", action="store_true", default=False)  # use transforms method in MoCo

    args = parser.parse_args()

    assert hasattr(args, "data") and isinstance(args.data, str)
    if args.data == "cifar10":
        args.num_classes = 10
    elif args.data == "cifar100":
        args.num_classes = 100
    else:
        args.num_classes = -1
    logger.info(args)

    set_seed(int(args.seed))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.gpu = isinstance(args.cuda, str) and len(args.cuda) > 0 and torch.cuda.is_available()

    # if verbose:
    logger.info("torch.__version__:\n", torch.__version__)
    logger.info("torch.version.cuda:\n", torch.version.cuda)
    logger.info("torch.backends.cudnn.version():\n", cudnn.version())
    logger.info("torch.cuda.is_available():\n", torch.cuda.is_available())
    logger.info("torch.cuda.device_count():\n", torch.cuda.device_count())
    logger.info("torch.cuda.current_device():\n", torch.cuda.current_device())
    logger.info("torch.cuda.get_device_name(0):\n", torch.cuda.get_device_name(0))
    logger.info("torch.cuda.get_arch_list():\n", torch.cuda.get_arch_list())

    has_cuda = torch.cuda.is_available()
    device = torch.device("cpu" if not has_cuda else "cuda")
    logger.info(f"has_cuda: {has_cuda}; device: {device}")

    args.device = device

    trainer = ImageClassificationTrainer(args)
    trainer.train()

    timer_end = time.process_time()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    return True


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    sys.exit(main())
