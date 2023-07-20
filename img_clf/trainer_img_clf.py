# -*- coding:utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from img_clf.config import SettingsCIFAR100
from img_clf.utils import get_network_classification, WarmUpLR, most_recent_folder, most_recent_weights, \
    last_epoch, best_acc_weights, get_dataloader_cifar


class ImageClassificationTrainer:

    def __init__(self, args):
        logging.basicConfig(
            format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

        self.args = args

        self.has_cuda = torch.cuda.is_available() and hasattr(args, "gpu") and args.gpu
        has_cuda = torch.cuda.is_available()
        device = torch.device("cpu" if not has_cuda else "cuda")
        self.device = device
        self.logger.info(f"has_cuda: {has_cuda}; device: {device}")

        long_tail = bool(args.long_tail)
        self.logger.info(f"long_tail: {long_tail}")

        adversarial = bool(args.adversarial)
        self.logger.info(f"adversarial: {adversarial}")

        # data loading and preprocessing
        if hasattr(args, "n_img_syn"):
            n_img_syn = int(args.n_img_syn)
        else:
            n_img_syn = 0

        if hasattr(args, "syn_type"):
            syn_type = str(args.syn_type)
            assert syn_type in ["glide_base", "glide_upsample", "gan_base"]
        else:
            syn_type = "glide_base"

        if hasattr(args, "adv_as_test"):
            adv_as_test = bool(args.adv_as_test)
        else:
            adv_as_test = False

        if hasattr(args, "extra_transforms"):
            extra_transforms = bool(args.extra_transforms)
        else:
            extra_transforms = False

        if args.data == "cifar100":
            settings = SettingsCIFAR100()

            if hasattr(args, "n_img_train"):
                n_img_train = int(args.n_img_train)
            else:
                n_img_train = -1
            self.training_loader, self.val_loader, self.test_loader = get_dataloader_cifar(
                settings.TRAIN_MEAN,
                settings.TRAIN_STD,
                ds_name="cifar100",
                num_workers=4,
                batch_size=args.b,
                shuffle=True,
                n_img=n_img_train,
                n_img_syn=n_img_syn,
                syn_type=syn_type,
                long_tail=long_tail,
                adversarial=adversarial,
                adv_as_test=adv_as_test,
                extra_transforms=extra_transforms
            )
        else:
            raise ValueError(f"invalid input parameter --data: {args.data}")

        self.net = get_network_classification(args)
        self.logger.info("total parameters", sum(x.numel() for x in self.net.parameters()))  # ResNet-101: 42,697,380
        self.net = self.net.to(self.device)

        if hasattr(args, "epoch") and int(args.epoch) > 0:
            settings.EPOCH = int(args.epoch)
            self.logger.info(">>> set settings.EPOCH:", settings.EPOCH)

        self.settings = settings

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        self.train_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=settings.MILESTONES, gamma=0.2)
        self.iter_per_epoch = len(self.training_loader)
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.iter_per_epoch * args.warm)

    def train(self):
        if self.args.resume:  # continue training
            recent_folder = most_recent_folder(
                os.path.join(self.settings.CHECKPOINT_DIR, self.args.net), fmt=self.settings.DATE_FORMAT)
            if not recent_folder:
                raise Exception("no recent folder were found")

            checkpoint_dir = os.path.join(self.settings.CHECKPOINT_DIR, self.args.net, recent_folder)
        else:
            recent_folder = ""
            checkpoint_dir = os.path.join(self.settings.CHECKPOINT_DIR, self.args.net, self.settings.TIME_NOW)

        # if not os.path.exists(self.settings.LOG_DIR):
        #     os.makedirs(self.settings.LOG_DIR, exist_ok=True)

        # create checkpoint folder to save model
        if not os.path.exists(checkpoint_dir) or not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, "{data}-{net}-{type}-{epoch}.pth")
        checkpoint_path_best = ""
        best_acc = 0.0
        best_res = []
        best_epoch = -1

        if self.args.resume:  # continue training
            best_weights = best_acc_weights(
                os.path.join(self.settings.CHECKPOINT_DIR, self.args.net, recent_folder))
            if best_weights:
                weights_path = os.path.join(
                    self.settings.CHECKPOINT_DIR, self.args.net, recent_folder, best_weights)
                self.logger.info("found best acc weights file:{}".format(weights_path))
                self.logger.info("load best training file to test acc...")
                if os.path.isfile(weights_path):
                    self.net.load_state_dict(torch.load(weights_path))
                    best_acc = self.eval_training(self.test_loader, epoch=0, tb=False)
                    checkpoint_path_best = weights_path
                    self.logger.info("best acc is {:0.2f}".format(best_acc))
                else:
                    self.logger.info("loading failed")

            recent_weights_file = most_recent_weights(
                os.path.join(self.settings.CHECKPOINT_DIR, self.args.net, recent_folder))
            if not recent_weights_file:
                raise Exception("no recent weights file were found")
            weights_path = os.path.join(
                self.settings.CHECKPOINT_DIR, self.args.net, recent_folder, recent_weights_file)
            self.logger.info("loading weights file {} to resume training.....".format(weights_path))
            if os.path.isfile(weights_path):
                self.net.load_state_dict(torch.load(weights_path))
            else:
                self.logger.info("loading failed")

            resume_epoch = last_epoch(
                os.path.join(self.settings.CHECKPOINT_DIR, self.args.net, recent_folder))
            self.logger.info("resume_epoch:", resume_epoch)
        else:
            resume_epoch = 0

        EVAL_GAP_VAL = 1
        # EVAL_GAP_TEST = 100

        self.logger.info(f">>> self.settings.EPOCH: {self.settings.EPOCH}")
        for epoch in range(1, self.settings.EPOCH + 1):
            self.logger.info(f"\n Training Epoch: {epoch}")

            if epoch > self.args.warm:
                # self.train_scheduler.step(epoch)
                self.train_scheduler.step()

            if self.args.resume:
                if epoch <= resume_epoch:
                    continue

            self._train(epoch)

            assert isinstance(self.val_loader, DataLoader)
            if epoch % EVAL_GAP_VAL == 0:
                acc_eval, err_1_eval, err_5_eval = self.eval_training(self.val_loader, epoch, tb=True)

                # start to save best-performance model after the second MILESTONE
                # if epoch > self.settings.MILESTONES[1] and best_acc < acc_eval:
                if best_acc < acc_eval:
                    ckpt_path = checkpoint_path.format(
                        data=self.args.data, net=self.args.net, type="best", epoch=epoch)
                    self.logger.info(f"saving weights file to {ckpt_path}")
                    torch.save(self.net.state_dict(), ckpt_path)  # save memory

                    best_epoch = epoch
                    best_acc = acc_eval
                    best_res = [acc_eval, err_1_eval, err_5_eval]
                    checkpoint_path_best_last = checkpoint_path_best
                    if os.path.isfile(checkpoint_path_best_last):
                        os.remove(checkpoint_path_best_last)
                    checkpoint_path_best = ckpt_path

                # if epoch % EVAL_GAP_TEST == 0 and os.path.isfile(checkpoint_path_best):
                #     cur_ckpt = self.net.state_dict()
                #     self.net.load_state_dict(torch.load(checkpoint_path_best))
                #     self.eval_training(self.test_loader, epoch, tb=True)
                #     self.net.load_state_dict(cur_ckpt)

        if os.path.isfile(checkpoint_path_best):
            self.logger.info(f"\ncheckpoint_path_best: {checkpoint_path_best}; best_res val: {best_res}")
            self.net.load_state_dict(torch.load(checkpoint_path_best))
            best_acc_val = self.eval_training(self.val_loader, epoch=best_epoch, tb=True)
            best_acc_test = self.eval_training(self.test_loader, epoch=best_epoch, tb=True)
            self.logger.info(
                f"\n>>> best at epoch {best_epoch}: best_acc_val is {best_acc_val}; best_acc_test is {best_acc_test}")
        else:
            self.logger.info(f"\n!!! no checkpoint_path_best !!! best_res val: {best_res}")

    def _train(self, epoch):
        start = time.time()
        self.net.train()

        dataloader_list = [self.training_loader]

        for d_loader in dataloader_list:
            for batch_index, (images, labels) in enumerate(d_loader):
                labels = labels.to(self.device)
                images = images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if epoch <= self.args.warm:
                    self.warmup_scheduler.step()

        finish = time.time()

        self.logger.info("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))

    @torch.no_grad()
    def eval_training(self, eval_loader, epoch=0, tb=True):
        start = time.time()
        self.net.eval()
        if eval_loader == self.test_loader:
            cur_loader = "test_loader"
            self.logger.info("Evaluating Network..... (test_loader)")
        elif eval_loader == self.val_loader:
            cur_loader = "val_loader"
            self.logger.info("Evaluating Network..... (val_loader)")
        else:
            raise ValueError(f"error test_loader")

        test_loss = 0.0
        correct = 0.0
        correct_1 = 0.0  # the number of top-1 error samples
        correct_5 = 0.0  # the number of top-5 error samples

        for (images, labels) in eval_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.net(images)
            loss = self.loss_function(outputs, labels)
            test_loss += loss.item()

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

            _, preds = outputs.topk(5, 1, largest=True, sorted=True)
            labels = labels.view(labels.size(0), -1).expand_as(preds)
            _correct = preds.eq(labels).float()
            correct_1 += _correct[:, :1].sum()
            correct_5 += _correct[:, :5].sum()

        n_test = len(eval_loader.dataset)
        acc = correct.float() / n_test
        err_1 = 1 - correct_1 / n_test
        err_5 = 1 - correct_5 / n_test

        if tb:
            self.logger.info(
                "{}: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Top 1 err: {:.4f}, Top 5 err: {:.4f}, "
                "Parameter numbers: {}, Time consumed:{:.2f}s".format(
                    cur_loader, epoch, test_loss / n_test, acc, err_1, err_5,
                    sum(p.numel() for p in self.net.parameters()), time.time() - start
                ))

        return acc, err_1, err_5
