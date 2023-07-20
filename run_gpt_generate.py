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
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, GPT2Model

from img_clf.utils import set_seed, get_prompts


def gpt_generate(syn_data: str, ckpt_epoch: int = 4):
    """generate sentences from labels/prompts by the GPT2 model
    generate description sentences from CIFAR-100, Office-31, Office-Home label text
        using the text-to-text model (GPT2) that fine-tuned on COCO captions for 5 epochs,
    and then input these descriptions to the text-to-image model (GLIDE) to generate images.
    """
    logger.info(f"\nGenerate Sentence Descriptions for {syn_data} by fine-tuned GPT-2\n")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    logger.info(f">>> tokenizer.all_special_tokens (before): {tokenizer.all_special_tokens}")
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
    if tokenizer.cls_token is None:
        tokenizer.cls_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.mask_token = tokenizer.eos_token
    logger.info(f">>> tokenizer.all_special_tokens (after): {tokenizer.all_special_tokens}")

    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    # model.to(device)
    # model.train()
    configuration = GPT2Config()
    model = GPT2LMHeadModel(configuration)

    gpt_dir = os.path.join("./data/gpt/")
    os.makedirs(gpt_dir, exist_ok=True)

    model_path = os.path.join(gpt_dir, f"gpt2_coco_ckpt_{ckpt_epoch}.pt")
    assert os.path.isfile(model_path), "Please run: `python run_gpt_finetune.py`"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    save_sent_dir = os.path.join(gpt_dir, f"gpt2_sent_{syn_data}/")
    if not os.path.isdir(save_sent_dir):
        os.makedirs(save_sent_dir, exist_ok=True)

    if syn_data == "coco_cap":
        raise ValueError(f"Do NOT generate sentences using original COCO captions")
    elif syn_data == "coco_cap_ner_gpt_sent":
        raise ValueError(f"Please use `--syn_data coco_cap_ner` to generate {syn_data}")
    else:
        prompts = get_prompts(syn_data)
    logger.info(f"len(prompts) = {len(prompts)}")

    x_set = sorted(prompts)
    x_set = [x.strip().replace("_", " ").replace("\t", " ") for x in x_set]

    sent_list = []
    for x in x_set:
        tokens = tokenizer.encode(x, return_tensors="pt")
        tokens = tokens.to(device)

        with torch.no_grad():
            outputs = model.generate(tokens,
                                     pad_token_id=tokenizer.pad_token_id,
                                     bos_token_id=tokenizer.bos_token_id,
                                     eos_token_id=tokenizer.eos_token_id,
                                     # max_length=10000,
                                     # max_length=50,
                                     max_length=20,
                                     num_beams=5,
                                     no_repeat_ngram_size=2,
                                     num_return_sequences=n_sentences,
                                     # temperature=0.7,
                                     # do_sample=True,
                                     # top_k=50,
                                     # top_p=0.95,
                                     early_stopping=True)

        for idx in range(n_sentences):
            sent = tokenizer.decode(outputs[idx], skip_special_tokens=True)
            sent = x + "\t" + sent
            logger.info(sent)
            sent_list.append(sent + "\n")

    sent_filepath = os.path.join(save_sent_dir, f"gpt2_generation_epoch_{ckpt_epoch}.txt")
    with open(sent_filepath, "a") as f_out:
        f_out.writelines(sent_list)


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
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--syn_data", type=str, default="coco_cap_ner",
                        choices=["cifar100", "office_31", "office_home",
                                 "coco_cap", "coco_cap_ner", "coco_cap_ner_gpt_sent"])
    parser.add_argument("--n_img", type=int, default=1, help="the number of sentences to be generated for each label")
    # parser.add_argument("--n_img", type=int, default=100)
    # parser.add_argument("--n_img", type=int, default=500)

    args = parser.parse_args()
    logger.info(args)

    set_seed(int(args.seed))

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

    has_cuda = torch.cuda.is_available()
    device = torch.device("cpu" if not has_cuda else "cuda")
    logger.info(f"has_cuda: {has_cuda}; device: {device}")

    n_sentences = int(args.n_sent)  # the number of sentences that needed to be generated from each label/concept

    gpt_generate(str(args.syn_data), ckpt_epoch=4)

    timer_end = time.process_time()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
