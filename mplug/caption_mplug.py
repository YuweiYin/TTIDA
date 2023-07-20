#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import time
import datetime
import random
import logging
import argparse
import json
import yaml
# import ruamel_yaml as yaml
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from mplug.language_evaluation import CocoEvaluator

from mplug.models.model_caption_mplug import MPLUG
from mplug.models.vit import resize_pos_embed
from mplug.models.tokenization_bert import BertTokenizer

import mplug.utils as utils
from mplug.dataset.utils import save_result
from mplug.dataset import create_dataset, create_sampler, create_loader, coco_collate_fn

from mplug.scheduler import create_scheduler
from mplug.optim import create_optimizer, create_two_optimizer

from img_clf.utils import set_seed


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=False, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter("lr1", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
        metric_logger.add_meter("lr2", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
    else:
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    for i, (image, caption, object_labels, image_ids, gold_caption) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        if config["prompt"] != "":
            caption = [config["prompt"] + each + config["eos"] for each in caption]
        else:
            caption = [each + config["eos"] for each in caption]
        question_input = [config["bos"] + " " + each for each in object_labels]
        if i == 0:
            logger.info(question_input)
        caption = tokenizer(caption, padding="longest", truncation=True, max_length=args.max_input_length,
                            return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding="longest", truncation=True, max_length=args.max_input_length,
                                   return_tensors="pt").to(device)
        # question_input = caption.input_ids[0,0].repeat(caption.input_ids.size(0), 1)

        if epoch > 0 or not config["warm_up"]:
            alpha = config["alpha"]
        else:
            alpha = config["alpha"] * min(1, i / len(data_loader))

        loss = model(image, question_input, caption, train=True)
        if accum_steps > 1:
            loss = loss / accum_steps

        if do_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # logger.info("scaled loss: {}".format(str(scaled_loss)))
                scaled_loss.backward()
        else:
            loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

        del image, question_input, caption, loss

        # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Generate Captioning test result:"
    print_freq = 50

    result = []

    # limit = 10  # for debugging
    # answer_input = None
    for n, (image, caption, object_labels, image_ids, gold_caption) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        caption = [each + config["eos"] for each in caption]
        question_input = [config["bos"] + " " + each for each in object_labels]
        caption = tokenizer(caption, padding="longest", truncation=True, max_length=args.max_input_length,
                            return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding="longest", truncation=True, max_length=args.max_input_length,
                                   return_tensors="pt").to(device)
        topk_ids, topk_probs = model(image, question_input, caption, train=False)

        for image_id, topk_id, topk_prob, gold_caption_list in zip(image_ids, topk_ids, topk_probs, gold_caption):
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            result.append({"question_id": image_id, "pred_caption": ans, "gold_caption": gold_caption_list})

        # if n == limit:  # for debugging
        #     break

    return result


@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = "Evaluation:"
    print_freq = 50
    predicts = []
    answers = []
    answer_input = None
    for n, (image, caption, image_ids, gold_caption) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        caption = [each + config["eos"] for each in caption]
        question_input = [config["bos"]] * len(caption)
        caption = tokenizer(caption, padding="longest", truncation=True, max_length=args.max_input_length,
                            return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding="longest", truncation=True, max_length=args.max_input_length,
                                   return_tensors="pt").to(device)

        for i in range(len(gold_caption)):
            predicts.append(gold_caption[i][0])
            answers.append(gold_caption[i])
        # {"Bleu_1": 0.9999999999863945, "Bleu_2": 0.9999999999859791, "Bleu_3": 0.9999999999854866,
        # "Bleu_4": 0.999999999984889, "METEOR": 1.0, "ROUGE_L": 1.0,
        # "CIDEr": 2.7246232035629268, "SPICE": 0.40389416048620613}
        result = cal_metric(predicts, answers)
        metric_logger.meters["Bleu_1"].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters["Bleu_2"].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters["Bleu_3"].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters["Bleu_4"].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters["Bleu_1"].update(result["Bleu_1"], n=image.size(0))

    # gather the stats from all processes
    torch.cuda.empty_cache()
    # metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def cal_metric(result_file):
    result_list = json.load(open(result_file, "r"))
    predicts = []
    answers = []
    for each in result_list:
        predicts.append(each["pred_caption"])
        answers.append(each["gold_caption"])
    # evaluator = language_evaluation.CocoEvaluator(verbose=False)
    # coco_types = ["BLEU", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
    # TODO: "METEOR": BrokenPipeError: [Errno 32] Broken pipe
    # TODO: "SPICE": subprocess.CalledProcessError
    coco_types = ["BLEU", "ROUGE_L", "CIDEr"]
    evaluator = CocoEvaluator(verbose=False, coco_types=coco_types)
    results = evaluator.run_evaluation(predicts, answers)
    logger.info(len(result_list), results)
    return results


def main(args, config):
    utils.init_distributed_mode(args)

    # device = torch.device(args.device)
    has_cuda = torch.cuda.is_available()
    logger.info("has_cuda:", has_cuda)
    # device = torch.device("cpu" if not has_cuda else f"cuda:{args.cuda}")
    device = torch.device("cpu" if not has_cuda else "cuda")
    # device = torch.device("cpu")
    logger.info("device:", device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]

    # Dataset
    logger.info("Creating Captioning datasets")
    datasets = create_dataset("coco", config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(
        datasets, samplers, batch_size=[
            config["batch_size_train"], config["batch_size_test"], config["batch_size_test"]],
        num_workers=[8, 8, 8], is_trains=[True, False, False],
        collate_fns=[coco_collate_fn, coco_collate_fn, coco_collate_fn])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    # Model
    logger.info("Creating model")
    model = MPLUG(config=config, tokenizer=tokenizer)
    model = model.to(device)

    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config["optimizer"])
        if not hasattr(arg_opt, "lr"):
            arg_opt.lr = args.lr
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config["optimizer"])
        optimizer = create_two_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.do_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        try:
            state_dict = checkpoint["model"]
        except:
            state_dict = checkpoint["module"]

        # reshape positional embedding to accommodate for image resolution change
        if config["clip_name"] == "ViT-B-16":
            num_patches = int(config["image_res"] * config["image_res"] / (16 * 16))
        elif config["clip_name"] == "ViT-L-14":
            num_patches = int(config["image_res"] * config["image_res"] / (14 * 14))
        else:
            raise ValueError(f"available config['clip_name'] is in ['ViT-B-16', 'ViT-L-14']")
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
        pos_embed = resize_pos_embed(state_dict["visual_encoder.visual.positional_embedding"].unsqueeze(0),
                                     pos_embed.unsqueeze(0))
        state_dict["visual_encoder.visual.positional_embedding"] = pos_embed

        if not args.evaluate:
            for key in list(state_dict.keys()):
                if ("fusion" in key or "bert" in key) and "decode" not in key:
                    encoder_key = key.replace("fusion.", "").replace("bert.", "")
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("load checkpoint from %s" % args.checkpoint)
        logger.info(msg)

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        import apex
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        model_without_ddp = model.module

    logger.info("Start training")
    start_time = time.time()

    captioning_result = evaluation(model, test_loader, tokenizer, device, config)
    result_file = save_result(captioning_result, args.result_dir, "captioning_result_epoch_first")
    if utils.is_main_process():
        result = cal_metric(result_file)
    # dist.barrier()

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(
                model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)

        if args.evaluate:
            break

        captioning_result = evaluation(model, test_loader, tokenizer, device, config)
        result_file = save_result(captioning_result, args.result_dir, f"captioning_result_epoch_{epoch}")
        if utils.is_main_process():
            result = cal_metric(result_file)
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch, }
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            torch.save({
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
            }, os.path.join(args.output_dir, f"checkpoint_{epoch}.pth"))

        # dist.barrier()

    captioning_result = evaluation(model, test_loader, tokenizer, device, config)
    result_file = save_result(captioning_result, args.result_dir, f"captioning_result_epoch_last")
    if utils.is_main_process():
        result = cal_metric(result_file)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    timer_start = time.process_time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42, help="random seed for all modules {7, 17, 42}")
    parser.add_argument("--bsz_train", type=int, default=32)
    parser.add_argument("--bsz_test", type=int, default=32)

    parser.add_argument("--n_img", type=int, default=-1, help="the number of original images. -1: all")
    parser.add_argument("--n_img_syn", type=int, default=0, help="the number of synthetic images. -1: all")
    # parser.add_argument("--n_img_syn", type=int, default=1000, help="the number of synthetic images. -1: all")
    parser.add_argument("--syn_type", type=str, default="base", help="base 64x64 or upsample 256x256")

    # # VQA
    # parser.add_argument("--config", default="./configs/VQA.yaml")
    # parser.add_argument("--checkpoint", default="")
    # parser.add_argument("--output_dir", default="output/vqa")
    # parser.add_argument("--evaluate", action="store_true")
    # parser.add_argument("--text_encoder", default="bert-base-uncased")
    # parser.add_argument("--text_decoder", default="bert-base-uncased")
    # parser.add_argument("--device", default="cuda")
    # parser.add_argument("--seed", default=42, type=int)
    # parser.add_argument("--min_length", default=1, type=int)
    # parser.add_argument("--lr", default=2e-5, type=float)
    # parser.add_argument("--max_length", default=10, type=int)
    # parser.add_argument("--max_input_length", default=25, type=int)
    # parser.add_argument("--beam_size", default=5, type=int)
    # parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    # parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    # parser.add_argument("--distributed", default=True, type=bool)
    # parser.add_argument("--do_two_optim", action="store_true")
    # parser.add_argument("--add_object", action="store_true")
    # parser.add_argument("--do_amp", action="store_true")
    # parser.add_argument("--no_init_decode", action="store_true")
    # parser.add_argument("--do_accum", action="store_true")
    # parser.add_argument("--accum_steps", default=4, type=int)

    # Image Captioning
    parser.add_argument("--config", default="./configs/caption_mplug_base.yaml")
    parser.add_argument("--checkpoint", default="./ckpt/coco_caption_mplug_base_1e-5.pth")
    parser.add_argument("--output_dir", default="output/coco_caption_mplug_base_1e-5")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--text_decoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--min_length", default=1, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--max_length", default=10, type=int)
    parser.add_argument("--max_input_length", default=25, type=int)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--do_two_optim", action="store_true")
    parser.add_argument("--add_object", action="store_true")
    parser.add_argument("--do_amp", action="store_true")
    parser.add_argument("--no_init_decode", action="store_true")
    parser.add_argument("--do_accum", action="store_true")
    parser.add_argument("--accum_steps", default=4, type=int)

    args = parser.parse_args()

    set_seed(int(args.seed))

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, "result")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    if args.bsz_train > 0:
        config["batch_size_train"] = args.bsz_train
    if args.bsz_test > 0:
        config["batch_size_test"] = args.bsz_test

    config["n_img"] = int(args.n_img)
    config["n_img_syn"] = int(args.n_img_syn)
    config["syn_type"] = str(args.syn_type)

    config["min_length"] = int(args.min_length)
    config["max_length"] = int(args.max_length)
    config["add_object"] = args.add_object
    config["beam_size"] = int(args.beam_size)
    # config["optimizer"]["lr"] = args.lr
    # config["schedular"]["lr"] = args.lr
    config["text_encoder"] = args.text_encoder
    config["text_decoder"] = args.text_decoder

    config["optimizer"]["lr"] = float(config["optimizer"]["lr"])
    config["optimizer"]["lr1"] = float(config["optimizer"]["lr1"])
    config["optimizer"]["lr2"] = float(config["optimizer"]["lr2"])
    config["optimizer"]["weight_decay"] = float(config["optimizer"]["weight_decay"])

    config["schedular"]["lr"] = float(config["schedular"]["lr"])
    config["schedular"]["min_lr"] = float(config["schedular"]["min_lr"])
    config["schedular"]["warmup_lr"] = float(config["schedular"]["warmup_lr"])
    config["schedular"]["decay_rate"] = float(config["schedular"]["decay_rate"])  # may be int type
    config["schedular"]["epochs"] = int(config["schedular"]["epochs"])
    config["schedular"]["warmup_epochs"] = int(config["schedular"]["warmup_epochs"])
    config["schedular"]["cooldown_epochs"] = int(config["schedular"]["cooldown_epochs"])

    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    logger.info(args)
    logger.info(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # if verbose:
    logger.info("torch.__version__:\n", torch.__version__)
    logger.info("torch.version.cuda:\n", torch.version.cuda)
    logger.info("torch.backends.cudnn.version():\n", torch.backends.cudnn.version())
    logger.info("torch.cuda.is_available():\n", torch.cuda.is_available())
    logger.info("torch.cuda.device_count():\n", torch.cuda.device_count())
    logger.info("torch.cuda.current_device():\n", torch.cuda.current_device())
    logger.info("torch.cuda.get_device_name(0):\n", torch.cuda.get_device_name(0))
    logger.info("torch.cuda.get_arch_list():\n", torch.cuda.get_arch_list())

    main(args, config)

    timer_end = time.process_time()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
