#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import argparse

from cdtrans.config import cfg
from cdtrans.datasets import make_dataloader
from cdtrans.model import make_model
from cdtrans.processor import do_inference, do_inference_uda
from cdtrans.utils.logger import setup_logger

from img_clf.utils import set_seed


if __name__ == "__main__":
    timer_start = time.process_time()

    parser = argparse.ArgumentParser(description="ReID Baseline Training")

    parser.add_argument("--seed", type=int, default=42, help="random seed for all modules {7, 17, 42}")

    parser.add_argument("--config_file", default="./uda.yml", help="path to config file", type=str)
    # parser.add_argument("--config_file", default="./pretrain.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

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

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    model = "deit_base"
    # model = "deit_small"

    set_seed(int(args.seed))

    data = args.data
    source_domain = args.source_domain
    target_domain = args.target_domain

    # cfg.MODEL.DEVICE_ID = "0"
    cfg.MODEL.DEVICE_ID = "1"
    cfg.MODEL.Transformer_TYPE = "uda_vit_base_patch16_224_TransReID"  # deit_base
    # cfg.MODEL.Transformer_TYPE = "uda_vit_small_patch16_224_TransReID"  # deit_small
    cfg.OUTPUT_DIR = f"./log/uda/{model}_test/{data}/{source_domain}_to_{target_domain}/"
    # cfg.MODEL.PRETRAIN_PATH = f"./log/pretrain/{model}/{data}/{source_domain}/transformer_10.pth"
    # cfg.MODEL.PRETRAIN_PATH = "./data/pretrainModel/deit_base_distilled_patch16_224-df68dfff.pth"
    cfg.MODEL.PRETRAIN_PATH = os.path.join(f"./data/pretrainModel/{data}/{source_domain}_to_{target_domain}/" +
                                           "office_uda_a2d_vit_base_768.pth")
    cfg.DATASETS.ROOT_TRAIN_DIR = f"./data/{data}/{source_domain}_list.txt"
    cfg.DATASETS.ROOT_TRAIN_DIR2 = f"./data/{data}/{target_domain}_list.txt"
    cfg.DATASETS.ROOT_TEST_DIR = f"./data/{data}/{target_domain}_list.txt"
    cfg.DATASETS.NAMES = "Office-31"
    cfg.DATASETS.NAMES2 = "Office-31"
    cfg.SOLVER.LOG_PERIOD = 10

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if os.path.isfile(args.config_file):
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    if cfg.MODEL.UDA_STAGE == "UDA":
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, \
            view_num, train_loader1, train_loader2, img_num1, img_num2, s_dataset, t_dataset = make_dataloader(cfg)
    else:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, \
            view_num = make_dataloader(cfg)

    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param_finetune(cfg.TEST.WEIGHT)
    if cfg.MODEL.UDA_STAGE == "UDA":
        do_inference_uda(cfg, model, val_loader, num_query)
    else:
        do_inference(cfg, model, val_loader, num_query)

    timer_end = time.process_time()
    print("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))
