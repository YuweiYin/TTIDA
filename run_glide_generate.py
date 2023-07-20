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
from PIL import Image

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

from img_clf.utils import set_seed, get_prompts


def generate_images(syn_data: str, n_img: int = 1, batch_size: int = 1, resize_img_dim: int = -1):
    """generate images from text prompts by the text-to-image model GLIDE"""
    logger.info(f"\nGenerate Synthetic Images for {syn_data} by GLIDE\n")

    # get prompts
    prompt_list = get_prompts(syn_data)
    # prompt_to_idx = {prompt: i for i, prompt in enumerate(prompt_list)}

    has_cuda = torch.cuda.is_available()
    device = torch.device("cpu" if not has_cuda else "cuda")
    logger.info(f"has_cuda: {has_cuda}; device: {device}")

    # Create base model
    options = model_and_diffusion_defaults()
    options["use_fp16"] = has_cuda
    options["timestep_respacing"] = "100"  # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint("base", device))
    logger.info("total base parameters", sum(x.numel() for x in model.parameters()))  # 385,030,726

    # Create upsampler model
    options_up = model_and_diffusion_defaults_upsampler()
    options_up["use_fp16"] = has_cuda
    options_up["timestep_respacing"] = "fast27"  # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint("upsample", device))
    logger.info("total upsampler parameters", sum(x.numel() for x in model_up.parameters()))  # 398,361,286

    # Sampling parameters
    guidance_scale = 3.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    # Create a classifier-free guidance sampling function
    def model_fn_base(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def save_image(output, save_dir: str, start_img_id: int, cur_idx: int):
        origin_shape = output.shape  # shape (N, C, H, W): (batch_size, 3, 64, 64)
        # scale and reshape
        output = ((output + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()  # shape (1, 3, H, W)
        output = output.permute(2, 0, 3, 1).reshape([origin_shape[2], -1, 3])  # shape (H, W, 3)
        img = Image.fromarray(output.numpy())
        if isinstance(resize_img_dim, int) and resize_img_dim > 0:
            img = img.resize((resize_img_dim, resize_img_dim))

        # img = Image.fromarray(output.numpy())
        # img.show()

        img_path = os.path.join(save_dir, f"{start_img_id + cur_idx}.png")
        img.save(fp=img_path)

    assert isinstance(n_img, int) and n_img >= 1

    total_cnt = 0
    for prompt_idx, prompt in enumerate(prompt_list):
        logger.info(f"\nprompt {prompt_idx + 1}: {prompt}")

        img_prompt_dir_upsample = os.path.join(f"./data/glide/glide_image_upsample/{syn_data}/", prompt)
        if not os.path.isdir(img_prompt_dir_upsample):
            os.makedirs(img_prompt_dir_upsample, exist_ok=True)

        filename_list_upsample = os.listdir(img_prompt_dir_upsample)
        if len(filename_list_upsample) > 0:
            filename_list_upsample = [int(fn[:-4]) for fn in filename_list_upsample]  # replace(".png", "")
            filename_list_upsample.sort()
            start_img_id_upsample = filename_list_upsample[-1] + 1
        else:
            start_img_id_upsample = 0

        for idx in range(n_img):
            ##############################
            # Sample from the base model #
            ##############################

            # Create the text tokens to feed to the model.
            tokens = model.tokenizer.encode(prompt)
            tokens, mask = model.tokenizer.padded_tokens_and_mask(
                tokens, options["text_ctx"]
            )

            # Create the classifier-free guidance tokens (empty)
            full_batch_size = batch_size * 2
            uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
                [], options["text_ctx"]
            )

            # Pack the tokens together into model kwargs.
            model_kwargs = dict(
                tokens=torch.tensor(
                    [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
                ),
                mask=torch.tensor(
                    [mask] * batch_size + [uncond_mask] * batch_size,
                    dtype=torch.bool,
                    device=device,
                ),
            )

            # Sample from the base model.
            model.del_cache()
            samples_base = diffusion.p_sample_loop(
                model_fn_base,
                (full_batch_size, 3, options["image_size"], options["image_size"]),
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=None,
            )[:batch_size]
            model.del_cache()

            # save_image(samples_base, img_prompt_dir_base, start_img_id_base, idx)

            ##############################
            # Upsample the 64x64 samples #
            ##############################

            tokens = model_up.tokenizer.encode(prompt)
            tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
                tokens, options_up["text_ctx"]
            )

            # Create the model conditioning dict.
            model_kwargs = dict(
                # Low-res image to upsample.
                low_res=((samples_base + 1) * 127.5).round() / 127.5 - 1,

                # Text tokens
                tokens=torch.tensor(
                    [tokens] * batch_size, device=device
                ),
                mask=torch.tensor(
                    [mask] * batch_size,
                    dtype=torch.bool,
                    device=device,
                ),
            )

            # Sample from the base model.
            model_up.del_cache()
            up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
            samples_up = diffusion_up.ddim_sample_loop(
                model_up,
                up_shape,
                noise=torch.randn(up_shape, device=device) * upsample_temp,
                device=device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=None,
            )[:batch_size]
            model_up.del_cache()

            save_image(samples_up, img_prompt_dir_upsample, start_img_id_upsample, idx)

            total_cnt += 1
            logger.info(f"[{total_cnt}]: {idx + 1} / {n_img}; [Prompt]: {prompt}")


def main() -> bool:
    timer_start = time.process_time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42, help="random seed for all modules {7, 17, 42}")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resize_img_dim", type=int, default=-1)
    parser.add_argument("--finetune_epoch", type=int, default=5)
    parser.add_argument("--syn_data", type=str, default="cifar100",
                        choices=["cifar100", "office_31", "office_home",
                                 "coco_cap", "coco_cap_ner", "coco_cap_ner_gpt_sent"])
    parser.add_argument("--n_img", type=int, default=1, help="the number of images to be generated for each prompt")
    # parser.add_argument("--n_img", type=int, default=100)
    # parser.add_argument("--n_img", type=int, default=500)

    args = parser.parse_args()
    logger.info(args)

    set_seed(int(args.seed))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # verbose = bool(args.verbose)
    # if verbose:
    logger.info("torch.__version__:\n", torch.__version__)
    logger.info("torch.version.cuda:\n", torch.version.cuda)
    logger.info("torch.backends.cudnn.version():\n", torch.backends.cudnn.version())
    logger.info("torch.cuda.is_available():\n", torch.cuda.is_available())
    logger.info("torch.cuda.device_count():\n", torch.cuda.device_count())
    logger.info("torch.cuda.current_device():\n", torch.cuda.current_device())
    logger.info("torch.cuda.get_device_name(0):\n", torch.cuda.get_device_name(0))
    logger.info("torch.cuda.get_arch_list():\n", torch.cuda.get_arch_list())

    # finetune_glide(int(args.finetune_epoch))
    generate_images(str(args.syn_data), n_img=int(args.n_img),
                    batch_size=int(args.batch_size),
                    resize_img_dim=int(args.resize_img_dim))

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
