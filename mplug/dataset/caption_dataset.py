# -*- coding:utf-8 -*-

import os
from io import BytesIO
import json
import time
import random
# import numpy as np
# import logging

import oss2
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

from mplug.dataset.utils import pre_caption

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def decode_int32(ann):
    ann = str(ann)
    server = str(int(ann[-1]) + 1)
    id_ = "0" * (9 - len(ann[:-1])) + ann[:-1]
    assert len(id_) == 9
    ann = server + "/" + id_
    return ann


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


class nocaps_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True,
                 add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.add_object = add_object

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        """ 
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
        """
        image_id = ann['img_id']
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    # file_str = bucket.get_object(file_path)
                    # file_buf = io.BytesIO()
                    # file_buf.write(file_str.read())
                    # file_buf.seek(0)
                    # file_buf = BytesIO(bucket.get_object(file_path).read())
                    # img_info = np.load(file_buf)
                    # file_buf.close()
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    # logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, image_id


class coco_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True,
                 add_object=False,
                 n_img: int = -1,  # the number of images per class. -1: all
                 n_img_syn: int = 0,  # the number of synthetic images per class. -1: all. 0: none.
                 ):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each["filename"]
            sentences = each["sentences"]
            filepath = each["filepath"]
            if filepath == "val2014":
                # file_root = "val2014_img"
                file_root = "val2014"
            elif filepath == "train2014":
                # file_root = "train2014_img"
                file_root = "train2014"
            elif filepath == "test2014":
                # file_root = "test2014_img"
                file_root = "test2014"
            else:
                file_root = filepath
            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for sent in sentences:
                caption = sent["raw"]
                gold_caption.append(caption.lower())
            if self.add_object:
                object_list = each["object_label"].split("&&")
                new_object_list = list(set(object_list))
                new_object_list.sort(key=object_list.index)
                object_label = " ".join(new_object_list)
            else:
                object_label = ""
            if is_train:
                for sent in sentences:
                    caption = sent["raw"].lower()
                    self.ann_new.append({"image": image_path, "caption": caption, "gold_caption": gold_caption,
                                         "object_label": object_label})
            else:
                self.ann_new.append(
                    {"image": image_path, "caption": sentences[0]["raw"].lower(), "gold_caption": gold_caption,
                     "object_label": object_label})

        if n_img > 0:
            self.ann_new = self.ann_new[:n_img]

        self.ann = self.ann_new
        del self.ann_new
        print("len(self.ann):", len(self.ann))  # len(self.ann): 591757

        if n_img_syn <= 0:
            return

        sent_root_dir = f"../data/gpt2_sentence_coco_ner/"
        if not os.path.isdir(sent_root_dir):
            os.makedirs(sent_root_dir, exist_ok=True)
        assert os.path.isdir(sent_root_dir)
        # ckpt_epoch = 0
        ckpt_epoch = 4
        sent_filepath = os.path.join(sent_root_dir, f"gpt2_generation_epoch_{ckpt_epoch}.txt")
        with open(sent_filepath, "r") as f_in:
            sent_list = f_in.readlines()
        sent_list = [sent.strip().split("\t") for sent in sent_list]

        root_syn = os.path.join(f"../data/glide_image_coco_cap_from_sent/NER/")
        assert os.path.isdir(root_syn)
        syn_filename_list = os.listdir(root_syn)
        random.shuffle(syn_filename_list)

        if n_img_syn > 0:
            syn_filename_list = syn_filename_list[:n_img_syn]
        else:
            return

        self.ann_syn = []
        for syn_filename in syn_filename_list:
            syn_idx = int(syn_filename.split("_")[0])
            ner, prompt = sent_list[syn_idx]
            # print(f"\nsentence: syn_idx: {syn_idx}; ner: {ner}; prompt: {prompt}")
            image_path = os.path.join(root_syn, syn_filename)
            assert os.path.isfile(image_path)
            caption = prompt.lower()
            gold_caption = [caption for _ in range(5)]
            object_label = ""
            self.ann_syn.append(
                {"image": image_path, "caption": caption, "gold_caption": gold_caption, "object_label": object_label}
            )

        print("len(self.ann_syn):", len(self.ann_syn))
        self.ann.extend(self.ann_syn)
        del self.ann_syn
        print("concatenated len(self.ann):", len(self.ann))

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1]
        object_label = ann['object_label']
        if self.read_local_data:
            if os.path.isfile(ann['image']):
                image_path = ann['image']
            else:
                image_path = os.path.join(self.root_path, ann['image'])
            # assert os.path.isfile(image_path)
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    # logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, caption, object_label, image_id, ann["gold_caption"]


class pretrain_dataset_4m(Dataset):
    def __init__(self, ann_file, transform, max_words=30, read_local_data=True, image_root="", epoch=None):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.image_root = image_root
        if not self.read_local_data:
            bucket_name = "xxxxx"
            auth = oss2.Auth("xxxxx", "xxxxxx")
            self.bucket = oss2.Bucket(auth, "xxxxx", bucket_name)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        if self.read_local_data:
            image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')
            image = self.transform(image)
        else:
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    # logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    time.sleep(0.1)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, caption
