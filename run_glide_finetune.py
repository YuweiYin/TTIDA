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
from glob import glob

import torch

from glide_finetune.glide_finetune import run_glide_finetune_epoch
from glide_finetune.glide_util import load_model
from glide_finetune.loader import TextImageDataset
from glide_finetune.wds_loader import glide_wds_loader

from img_clf.utils import set_seed


def run_glide_finetune(
        data_dir="./data/COCO_2015_Captioning/train2014/",
        batch_size=1,  # batch size of the training set
        learning_rate=1e-5,
        adam_weight_decay=0.0,
        side_x=64,
        side_y=64,
        resize_ratio=1.0,
        uncond_p=0.0,
        resume_ckpt="",
        checkpoints_dir="./ckpt/glide_finetune/",
        use_fp16=False,  # Tends to cause issues,not sure why as the paper states fp16 is stable.
        device=torch.device("cpu"),
        freeze_transformer=False,
        freeze_diffusion=False,
        project_name="glide_finetune",
        activation_checkpointing=False,
        use_captions=True,
        num_epochs=100,
        log_frequency=100,
        test_prompt="a group of skiers are preparing to ski down a mountain.",
        sample_bs=1,  # batch size for inference
        sample_gs=8.0,
        use_webdataset=False,
        image_key="jpg",
        caption_key="txt",
        enable_upsample=False,
        upsample_factor=4,
        image_to_upsample="low_res_face.png",
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir:
        checkpoints_dir = os.path.expanduser(checkpoints_dir)

    # Create the checkpoint/output directories
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Start wandb logging
    # logger.info("Wandb setup.")
    # wandb_run = wandb_setup(
    #     batch_size=batch_size,
    #     side_x=side_x,
    #     side_y=side_y,
    #     learning_rate=learning_rate,
    #     use_fp16=use_fp16,
    #     device=device,
    #     data_dir=data_dir,
    #     base_dir=checkpoints_dir,
    #     project_name=project_name,
    # )
    wandb_run = None

    # Model setup
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=resume_ckpt,
        use_fp16=use_fp16,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
        model_type="base" if not enable_upsample else "upsample",
    )
    glide_model.train()
    number_of_params = sum(x.numel() for x in glide_model.parameters())
    logger.info(f"Number of parameters: {number_of_params}")
    number_of_trainable_params = sum(
        x.numel() for x in glide_model.parameters() if x.requires_grad
    )
    logger.info(f"Trainable parameters: {number_of_trainable_params}")

    # Data setup
    logger.info("Loading data...")
    if use_webdataset:
        dataset = glide_wds_loader(
            urls=data_dir,
            caption_key=caption_key,
            image_key=image_key,
            enable_image=True,
            enable_text=use_captions,
            enable_upsample=enable_upsample,
            tokenizer=glide_model.tokenizer,
            ar_lower=0.5,
            ar_upper=2.0,
            min_original_height=side_x * upsample_factor,
            min_original_width=side_y * upsample_factor,
            upscale_factor=upsample_factor,
            nsfw_filter=True,
            similarity_threshold_upper=0.0,
            similarity_threshold_lower=0.5,
            words_to_skip=[],
            dataset_name="laion",  # "laion" or "alamy"
        )
    else:
        dataset = TextImageDataset(
            folder=data_dir,
            side_x=side_x,
            side_y=side_y,
            resize_ratio=resize_ratio,
            uncond_p=uncond_p,
            shuffle=True,
            tokenizer=glide_model.tokenizer,
            text_ctx_len=glide_options["text_ctx"],
            use_captions=use_captions,
            enable_glide_upsample=enable_upsample,
            upscale_factor=upsample_factor,
        )

    # Data loader setup
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not use_webdataset,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        [x for x in glide_model.parameters() if x.requires_grad],
        lr=learning_rate,
        weight_decay=adam_weight_decay,
    )

    # if we want to train the transformer, we need to back-propagate through the diffusion model.
    if not freeze_transformer:
        glide_model.out.requires_grad_(True)
        glide_model.input_blocks.requires_grad_(True)
        glide_model.middle_block.requires_grad_(True)
        glide_model.output_blocks.requires_grad_(True)

    # Training setup
    outputs_dir = "./outputs/glide_finetune_test/"
    os.makedirs(outputs_dir, exist_ok=True)

    existing_runs = [sub_dir for sub_dir in os.listdir(checkpoints_dir) if
                     os.path.isdir(os.path.join(checkpoints_dir, sub_dir))]
    existing_runs_int = []
    for x in existing_runs:
        try:
            existing_runs_int.append(int(x))
        except:
            logger.info("unexpected directory naming scheme")  # ignore
    existing_runs_int = sorted(existing_runs_int)
    next_run = 0 if len(existing_runs) == 0 else existing_runs_int[-1] + 1
    current_run_ckpt_dir = os.path.join(checkpoints_dir, str(next_run).zfill(4))

    os.makedirs(current_run_ckpt_dir, exist_ok=True)

    # for epoch in trange(num_epochs):
    for epoch in range(num_epochs):
        logger.info(f"\n>>> Training epoch: {epoch}")
        run_glide_finetune_epoch(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            optimizer=optimizer,
            dataloader=dataloader,
            prompt=test_prompt,
            sample_bs=sample_bs,
            sample_gs=sample_gs,
            checkpoints_dir=current_run_ckpt_dir,
            outputs_dir=outputs_dir,
            side_x=side_x,
            side_y=side_y,
            device=device,
            wandb_run=wandb_run,
            log_frequency=log_frequency,
            epoch=epoch,
            gradient_accumulation_steps=1,
            train_upsample=enable_upsample,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42, help="random seed for all modules {7, 17, 42}")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--data_dir", "-data", type=str, default="./data/COCO_2015_Captioning/train2014/")
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--adam_weight_decay", "-adam_wd", type=float, default=0.0)
    parser.add_argument("--side_x", "-x", type=int, default=64)
    parser.add_argument("--side_y", "-y", type=int, default=64)
    parser.add_argument("--resize_ratio", "-crop", type=float, default=0.8, help="Crop ratio")
    parser.add_argument("--uncond_p", "-p", type=float, default=0.2,
                        help="Probability of using the empty/unconditional token instead of a caption. "
                             "OpenAI used 0.2 for finetune.")
    parser.add_argument("--train_upsample", "-upsample", action="store_true",
                        help="Train the upsampling type of the model instead of the base model.")
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument("--freeze_transformer", "-fz_xt", action="store_true")
    parser.add_argument("--freeze_diffusion", "-fz_unet", action="store_true")
    parser.add_argument("--project_name", "-name", type=str, default="glide-finetune")
    parser.add_argument("--activation_checkpointing", "-grad_ckpt", action="store_true")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    parser.add_argument("--epochs", "-epochs", type=int, default=20)
    parser.add_argument("--test_prompt", "-prompt", type=str,
                        default="a group of skiers are preparing to ski down a mountain.")
    parser.add_argument("--test_batch_size", "-tbs", type=int, default=1,
                        help="Batch size used for model eval, not training.")
    parser.add_argument("--test_guidance_scale", "-tgs", type=float, default=4.0,
                        help="Guidance scale used during model eval, not training.")
    parser.add_argument("--use_webdataset", "-wds", action="store_true", help="Enables webdataset (tar) loading")
    parser.add_argument("--wds_image_key", "-wds_img", type=str, default="jpg",
                        help="A 'key' e.g. 'jpg' used to access the image in the webdataset")
    parser.add_argument("--wds_caption_key", "-wds_cap", type=str, default="txt",
                        help="A 'key' e.g. 'txt' used to access the caption in the webdataset")
    parser.add_argument("--wds_dataset_name", "-wds_name", type=str, default="laion",
                        help="Name of the webdataset to use (laion or alamy)")
    parser.add_argument("--seed", "-seed", type=int, default=17)  # default=0
    parser.add_argument("--cudnn_benchmark", "-cudnn", action="store_true",
                        help="Enable cudnn benchmarking. May improve performance. (may not)")
    parser.add_argument("--upscale_factor", "-upscale", type=int, default=4,
                        help="Upscale factor for training the upsampling model only")
    parser.add_argument("--image_to_upsample", "-lowres", type=str, default="low_res_face.png")

    parser.add_argument("--checkpoints_dir", "-ckpt", type=str, default="./ckpt/glide_finetune/")
    parser.add_argument("--resume_ckpt", "-resume", type=str, default="", help="Checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    timer_start = time.process_time()

    args = parse_args()
    logger.info(args)

    set_seed(int(args.seed))

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    has_cuda = torch.cuda.is_available()
    device = torch.device("cpu" if not has_cuda else "cuda")
    logger.info(f"has_cuda: {has_cuda}; device: {device}")

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

    if args.use_webdataset:
        data_dir = glob(os.path.join(args.data_dir, "*.tar"))  # webdataset uses tars
    else:
        data_dir = args.data_dir

    run_glide_finetune(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        adam_weight_decay=args.adam_weight_decay,
        side_x=args.side_x,
        side_y=args.side_y,
        resize_ratio=args.resize_ratio,
        uncond_p=args.uncond_p,
        resume_ckpt=args.resume_ckpt,
        checkpoints_dir=args.checkpoints_dir,
        use_fp16=args.use_fp16,
        device=device,
        log_frequency=args.log_frequency,
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        project_name=args.project_name,
        activation_checkpointing=args.activation_checkpointing,
        use_captions=args.use_captions,
        num_epochs=args.epochs,
        test_prompt=args.test_prompt,
        sample_bs=args.test_batch_size,
        sample_gs=args.test_guidance_scale,
        use_webdataset=args.use_webdataset,
        image_key=args.wds_image_key,
        caption_key=args.wds_caption_key,
        enable_upsample=args.train_upsample,
        upsample_factor=args.upscale_factor,
        image_to_upsample=args.image_to_upsample,
    )

    timer_end = time.process_time()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
