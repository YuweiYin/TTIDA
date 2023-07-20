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
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from img_clf.dataset import CocoCaptionNerForGptFinetune
from img_clf.utils import set_seed


def gpt_finetune():
    """The vanilla GPT-2 model can not generate good sentences with the class_name or concept input,
    so we need to find-tune the model on the training set of COCO captioning dataset"""

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

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=float(1e-3), weight_decay=float(5e-4))
    logger.info(optimizer)

    gpt_dir = os.path.join("./data/gpt/")
    coco_cap_ner = torch.load(os.path.join(gpt_dir, "coco_cap_ner.pt"))
    coco_ner = coco_cap_ner["coco_ner"]
    coco_cap = coco_cap_ner["coco_cap"]
    coco_ner = [[ner.strip().lower() for ner in ner_list] for ner_list in coco_ner]
    coco_ner = ["\t".join(ner_list) if len(ner_list) > 1 else ner_list[0] for ner_list in coco_ner]
    coco_cap = [cap.strip().lower() for cap in coco_cap]
    coco_cap = [cap[:-1] if cap[-1] == "." else cap for cap in coco_cap]  # get rid of the last "."
    assert isinstance(coco_ner, list) and isinstance(coco_cap, list) and len(coco_ner) == len(coco_cap)

    coco_cap_ner_dataset = CocoCaptionNerForGptFinetune(data_cap=coco_cap, data_ner=coco_ner)
    coco_cap_ner_dataloader = DataLoader(coco_cap_ner_dataset, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):
        logger.info(f"\n\nFine-tuning GPT-2: Epoch {epoch}")

        loss_log = []
        log_gap = 1000

        for batch_index, (captions, ner_list) in enumerate(coco_cap_ner_dataloader):
            assert len(captions) == len(ner_list)
            prompt_enc_list = []
            for caption, ner_array in zip(captions, ner_list):
                ner_array = ner_array.split("\t") if "\t" in ner_array else [ner_array]
                prompt = f"Write an image description of {ner_array[0]} with keywords including {ner_array[0]}"
                for ner in ner_array[1:]:
                    prompt += f", {ner}"
                prompt += f" : {caption} {tokenizer.eos_token}"
                prompt_enc = tokenizer.encode(prompt)
                prompt_enc_list.append(prompt_enc)

            max_len = max([len(p) for p in prompt_enc_list]) + 1
            # pad_tok = -100
            # pad_tok_id = tokenizer.encoder[tokenizer.pad_token]
            pad_tok_id = tokenizer.pad_token_id
            prompt_enc_pad_list = []
            for prompt_enc in prompt_enc_list:  # padding
                prompt_enc_pad_list.append(prompt_enc + [pad_tok_id] * (max_len - len(prompt_enc)))
            prompt_tensor = torch.tensor(prompt_enc_pad_list, dtype=torch.int64, device=device)

            outputs = model(prompt_tensor, labels=prompt_tensor)
            loss = outputs.loss  # Language modeling loss (for next-token prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % log_gap == 0:
                loss_log.append(f"epoch {epoch} - batch {batch_index}: loss: {loss}\n")

        # save model
        gpt_dir = os.path.join("./data/gpt/")
        os.makedirs(gpt_dir, exist_ok=True)

        save_model_path = os.path.join(gpt_dir, f"gpt2_coco_ckpt_{epoch}.pt")
        torch.save(model.state_dict(), save_model_path)

        # save log
        save_log_path = os.path.join(gpt_dir, f"gpt2_coco_loss_log_{epoch}.txt")
        with open(save_log_path, "w") as w_f:
            w_f.writelines(loss_log)


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=5)

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

    batch_size = int(args.batch_size)
    epochs = int(args.epoch)

    gpt_finetune()

    timer_end = time.process_time()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
