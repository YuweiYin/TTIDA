# -*- coding:utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import datetime


class SettingsCIFAR10:

    def __init__(self):
        self.TRAIN_MEAN = (0.485, 0.456, 0.406)
        self.TRAIN_STD = (0.229, 0.224, 0.225)

        self.VAL_MEAN = (0.485, 0.456, 0.406)
        self.VAL_STD = (0.229, 0.224, 0.225)

        self.TEST_MEAN = (0.485, 0.456, 0.406)
        self.TEST_STD = (0.229, 0.224, 0.225)

        self.EPOCH = 200  # total training epochs
        self.MILESTONES = [60, 120, 160]
        # self.EPOCH = 100  # total training epochs
        # self.MILESTONES = [20, 40, 60]
        # self.INIT_LR = 0.1  # initial learning rate

        self.DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
        self.TIME_NOW = datetime.datetime.now().strftime(self.DATE_FORMAT)  # time of we run the script

        self.LOG_DIR = 'log/log_run_cifar10'  # tensorboard log dir
        self.SAVE_EPOCH = 10  # save weights file per SAVE_EPOCH epoch
        self.CHECKPOINT_DIR = 'ckpt/ckpt_cifar10'  # directory to save weights file


class SettingsCIFAR100:

    def __init__(self):
        self.TRAIN_MEAN = (0.485, 0.456, 0.406)
        self.TRAIN_STD = (0.229, 0.224, 0.225)

        self.VAL_MEAN = (0.485, 0.456, 0.406)
        self.VAL_STD = (0.229, 0.224, 0.225)

        self.TEST_MEAN = (0.485, 0.456, 0.406)
        self.TEST_STD = (0.229, 0.224, 0.225)

        self.EPOCH = 200  # total training epochs
        self.MILESTONES = [60, 120, 160]
        # self.EPOCH = 100  # total training epochs
        # self.MILESTONES = [20, 40, 60]
        # self.INIT_LR = 0.1  # initial learning rate

        self.DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
        self.TIME_NOW = datetime.datetime.now().strftime(self.DATE_FORMAT)  # time of we run the script

        self.LOG_DIR = 'log/log_run_cifar100'  # tensorboard log dir
        self.SAVE_EPOCH = 10  # save weights file per SAVE_EPOCH epoch
        self.CHECKPOINT_DIR = 'ckpt/ckpt_cifar100'  # directory to save weights file
