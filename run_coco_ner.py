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

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

import corenlp
from pycocotools.coco import COCO
from img_clf.utils import set_seed


def main() -> bool:
    timer_start = time.process_time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed for all modules {7, 17, 42}")
    args = parser.parse_args()
    logger.info(args)

    set_seed(int(args.seed))

    # initialize COCO API for instance annotations
    coco_data_root = "./data/COCO_2015_Captioning/"
    coco_train_cap = COCO(os.path.join(coco_data_root, "annotations/captions_train2014.json"))

    # coco_train_ins = COCO(os.path.join(coco_data_root, "annotations/instances_train2014.json"))
    # coco_train_per = COCO(os.path.join(coco_data_root, "annotations/person_keypoints_train2014.json"))
    # coco_valid_cap = COCO(os.path.join(coco_data_root, "annotations/captions_val2014.json"))
    # coco_valid_ins = COCO(os.path.join(coco_data_root, "annotations/instances_val2014.json"))
    # coco_valid_per = COCO(os.path.join(coco_data_root, "annotations/person_keypoints_val2014.json"))

    def build_vocab():
        from img_clf.dataset import Vocabulary
        vocab = Vocabulary(vocab_from_file=False)

    def show_coco_info():
        coco_train_cap_ids = list(coco_train_cap.anns.keys())

        # pick a random image and obtain the corresponding URL
        ann_id = np.random.choice(coco_train_cap_ids)
        img_id = coco_train_cap.anns[ann_id]["image_id"]
        img = coco_train_cap.loadImgs(img_id)[0]
        url = img["coco_url"]

        # print URL and visualize corresponding image
        logger.info(url)
        show_img = io.imread(url)
        plt.axis("off")
        plt.imshow(show_img)
        plt.show()
        plt.savefig("./data/show_coco/show_coco.png")

        # load and display captions
        annIds = coco_train_cap.getAnnIds(imgIds=img["id"])
        anns = coco_train_cap.loadAnns(annIds)
        coco_train_cap.showAnns(anns)

    def extract_ner_from_coco_captions():
        coco_train_cap_ids = list(coco_train_cap.anns.keys())
        logger.info("len(coco_train_cap_ids):", len(coco_train_cap_ids))  # 414113

        # ner tags: LOCATION, ORGANIZATION, PERSON
        # DATE, LOCATION, MONEY, ORGANIZATION, PERCENT, PERSON, TIME
        # LOCATION, MISC, ORGANIZATION, PERSON
        ner_tags = {"LOCATION", "ORGANIZATION", "PERSON"}

        # coco_anns = []
        # for ann_id in coco_train_cap_ids:
        #     ann_dict = coco_train_cap.anns[ann_id]
        #     text = ann_dict["caption"]
        #     coco_anns.append(text)

        coco_ner = []
        coco_cap = []

        # with corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma ner depparse".split()) as client:
        with corenlp.CoreNLPClient(annotators=["ner"]) as client:
            for ann_id in coco_train_cap_ids:
                ann_dict = coco_train_cap.anns[ann_id]
                text = ann_dict["caption"]
                # logger.info(text)

                ann = client.annotate(text)
                # logger.info(type(ann))
                # logger.info(ann)
                # <class "doc.CoreNLP_pb2.Document">
                # text: "A very clean and well decorated empty bathroom"
                # sentence {
                #   token {
                #     word: "A"
                #     pos: "DT"
                #     value: "A"
                #     before: ""
                #     after: " "
                #     originalText: "A"
                #     ner: "O"
                #     lemma: "a"
                #     beginChar: 0
                #     endChar: 1
                #     tokenBeginIndex: 0
                #     tokenEndIndex: 1
                #     hasXmlContext: false
                #   }
                #   token {
                #     ...
                #   }
                #   ...
                # }

                sentence = ann.sentence[0]
                # assert corenlp.to_text(sentence) == text

                show_text = True
                ner_list = []
                for token in sentence.token:
                    token_word = token.word
                    token_pos = token.pos
                    token_ner = token.ner
                    if len(token_word) <= 0 or len(token_pos) <= 0:
                        continue
                    if token_ner in ner_tags or token_pos[0] == "N":
                        if show_text:
                            show_text = False
                            logger.info(text)
                        logger.info(f">>> WORD: {token.word}; POS: {token_pos}; NER: {token_ner}")
                        ner_list.append(token.word)
                        # gpt_ner.append(token.word)
                        # gpt_text.append(text)

                if len(ner_list) > 0:
                    coco_ner.append(ner_list)
                    coco_cap.append(text)

        logger.info("len(coco_ner):", len(coco_ner))  # 414096
        logger.info("len(coco_cap):", len(coco_cap))  # 414096
        assert len(coco_ner) == len(coco_cap)

        gpt_finetune_data = dict({})
        gpt_finetune_data["coco_ner"] = coco_ner
        gpt_finetune_data["coco_cap"] = coco_cap
        # gpt_finetune_data["coco_anns"] = coco_anns

        gpt_dir = os.path.join("./data/gpt/")
        os.makedirs(gpt_dir, exist_ok=True)

        import torch
        torch.save(gpt_finetune_data, os.path.join(gpt_dir, "coco_cap_ner.pt"))

    # build_vocab()
    # show_coco_info()
    extract_ner_from_coco_captions()

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
