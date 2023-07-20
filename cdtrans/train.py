#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import argparse
import torch

from cdtrans.config import cfg
from cdtrans.utils.logger import setup_logger
from cdtrans.datasets import make_dataloader
from cdtrans.model import make_model
from cdtrans.solver import make_optimizer, create_scheduler
from cdtrans.loss import make_loss
from cdtrans.processor import do_train_pretrain, do_train_uda

from img_clf.utils import set_seed


if __name__ == "__main__":
    timer_start = time.process_time()

    parser = argparse.ArgumentParser(description="ReID Baseline Training")

    # parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--cuda", type=str, default="1")

    parser.add_argument("--seed", type=int, default=42, help="random seed for all modules {7, 17, 42}")

    parser.add_argument("--epochs", type=int, default=40)

    parser.add_argument("--config_file", default="./uda.yml", help="path to config file", type=str)
    # parser.add_argument("--config_file", default="./pretrain.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)

    # Office-31: "amazon", "dslr", "webcam"
    parser.add_argument("--data", type=str, default="office_31", help="dataset name")
    parser.add_argument("--source_domain", type=str, default="amazon")  # 2817 images, 31 classes
    parser.add_argument("--target_domain", type=str, default="dslr")  # 498 images, 31 classes
    # # parser.add_argument("--target_domain", type=str, default="webcam")  # 795 images, 31 classes

    # Office-Home: "Art", "Clipart", "Product", "Real_World"
    # parser.add_argument("--data", type=str, default="office_home", help="dataset name")
    # parser.add_argument("--source_domain", type=str, default="Art")  # 2427 images, 65 classes
    # parser.add_argument("--target_domain", type=str, default="Clipart")  # 4365 images, 65 classes
    # # parser.add_argument("--source_domain", type=str, default="Product")  # 4439 images, 65 classes
    # # parser.add_argument("--target_domain", type=str, default="Real_World")  # 4357 images, 65 classes

    parser.add_argument("--n_img_syn", type=int, default=0, help="the number of synthetic images per class. -1: all")
    # parser.add_argument("--n_img_syn", type=int, default=-1, help="the number of synthetic images per class. -1: all")
    # parser.add_argument("--n_img_syn", type=int, default=10, help="the number of synthetic images per class. -1: all")
    parser.add_argument("--syn_type", type=str, default="base", help="base 64x64 or upsample 256x256")

    parser.add_argument("--pretrain", action="store_true", default=False)

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    model = "deit_base"
    # model = "deit_small"

    set_seed(int(args.seed))

    data = str(args.data)
    source_domain = str(args.source_domain)
    target_domain = str(args.target_domain)
    n_img_syn = int(args.n_img_syn)
    syn_type = str(args.syn_type)

    cfg.DATASETS.DATA = data
    cfg.DATASETS.SOURCE_DOMAIN = source_domain
    cfg.DATASETS.TARGET_DOMAIN = target_domain
    cfg.DATASETS.SYN_NUM = n_img_syn
    cfg.DATASETS.SYN_TYPE = syn_type
    # cfg.DATASETS.SYN_NUM = 0
    # cfg.DATASETS.SYN_NUM = 10
    # cfg.DATASETS.SYN_NUM = 50
    # cfg.DATASETS.SYN_NUM = 100
    # cfg.DATASETS.SYN_NUM = 200
    # cfg.DATASETS.SYN_NUM = 300
    # cfg.DATASETS.SYN_NUM = 400
    # cfg.DATASETS.SYN_NUM = 500

    pretrain = args.pretrain
    # pretrain = True
    cfg.MODEL.PRETRAIN_CHOICE = "pretrain" if pretrain else "train_from_scratch"

    cfg.MODEL.DEVICE_ID = args.cuda
    print(f"cfg.MODEL.DEVICE_ID: {cfg.MODEL.DEVICE_ID}")
    cfg.MODEL.Transformer_TYPE = "uda_vit_base_patch16_224_TransReID"  # deit_base
    # cfg.MODEL.Transformer_TYPE = "uda_vit_small_patch16_224_TransReID"  # deit_small
    cfg.OUTPUT_DIR = f"./log/uda/{model}_train/{data}/{source_domain}_to_{target_domain}/"

    cfg.SOLVER.MAX_EPOCHS = int(args.epochs)
    cfg.SOLVER.LOG_PERIOD = 10

    cfg.MODEL.PRETRAIN_PATH = "train_from_scratch"
    pretrain_dir = f"./data/pretrainModel/{data}/{source_domain}_to_{target_domain}/"
    assert os.path.isdir(pretrain_dir)
    ckpt_list = os.listdir(pretrain_dir)
    for ckpt in ckpt_list:
        if "base" in ckpt:  # get base model
            cfg.MODEL.PRETRAIN_PATH = os.path.join(pretrain_dir, ckpt)
    if os.path.isfile(cfg.MODEL.PRETRAIN_PATH):
        print("cfg.MODEL.PRETRAIN_CHOICE:", cfg.MODEL.PRETRAIN_CHOICE)
        print("cfg.MODEL.PRETRAIN_PATH:", cfg.MODEL.PRETRAIN_PATH)
    else:
        cfg.MODEL.PRETRAIN_CHOICE = "train_from_scratch"
        cfg.MODEL.PRETRAIN_PATH = "train_from_scratch"

    cfg.DATASETS.ROOT_TRAIN_DIR = f"./data/{data}/{source_domain}.txt"
    cfg.DATASETS.ROOT_TRAIN_DIR2 = f"./data/{data}/{target_domain}.txt"
    cfg.DATASETS.ROOT_TEST_DIR = f"./data/{data}/{target_domain}.txt"
    if data == "office_31":
        cfg.DATASETS.NAMES = "Office-31"
        cfg.DATASETS.NAMES2 = "Office-31"
    elif data == "office_home":
        cfg.DATASETS.NAMES = "Office-Home"
        cfg.DATASETS.NAMES2 = "Office-Home"
    else:
        raise ValueError(f"ValueError: source_domain == {source_domain}")

    cfg.freeze()

    # set_seed(cfg.SOLVER.SEED)
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    else:
        pass

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if os.path.isfile(args.config_file):
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    if cfg.MODEL.UDA_STAGE == "UDA":
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, \
            view_num, train_loader1, train_loader2, img_num1, img_num2, s_dataset, t_dataset = make_dataloader(cfg)
    else:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, \
            view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    if cfg.MODEL.UDA_STAGE == "UDA":
        do_train_uda(
            cfg,
            model, center_criterion,
            train_loader, train_loader1, train_loader2,
            img_num1, img_num2,
            val_loader,
            s_dataset, t_dataset,
            optimizer, optimizer_center,
            scheduler, loss_func,
            num_query, args.local_rank
        )
    else:
        print("pretrain train")
        do_train_pretrain(
            cfg,
            model, center_criterion,
            train_loader, val_loader,
            optimizer, optimizer_center,
            scheduler, loss_func,
            num_query, args.local_rank
        )

    timer_end = time.process_time()
    print("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))
