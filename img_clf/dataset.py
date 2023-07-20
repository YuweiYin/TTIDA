# -*- coding:utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import json
import random
from typing import Any, Callable, Optional, Tuple
from collections import Counter

import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from pycocotools.coco import COCO
import nltk


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
            self,
            root: str,
            train: str = "train",  # "train", "val", "test"
            test_size: float = 0.2,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            n_img: int = -1,  # the number of images per class. -1: all
            syn_data: str = "cifar100",
            n_img_syn: int = 0,  # the number of synthetic images per class. -1: all. 0: none.
            syn_type: str = "glide_base",  # glide_base 64x64 or glide_upsample 256x256 or gan_base 64x64
            long_tail: bool = False,  # use the long-tail subset for training or not
            adversarial: bool = False,  # add adversarial attacking images to the training set or not
            adv_as_test: bool = False,  # use adv images as test set or not
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._load_meta()

        self.train = train  # "train", "val", "test"
        self.img_dim = 32
        self.img_size = (self.img_dim, self.img_dim)  # RGB channels = 3

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        def _load_dav():
            _data_adv: Any = []
            _targets_adv = []

            _root_adv = os.path.join("./adversarial", f"{syn_data}/")
            assert os.path.isdir(_root_adv)
            _filename_list = os.listdir(_root_adv)

            for _filename in _filename_list:
                _class_name = "_".join(_filename.split("_")[:-1])

                _file_path = os.path.join(_root_adv, _filename)
                assert os.path.isfile(_file_path)
                _img = Image.open(_file_path)
                if _img.mode != "RGB":
                    _img = _img.convert("RGB")
                if _img.size != self.img_size:
                    _img = _img.resize(size=self.img_size)
                _data_adv.append(np.array(_img))
                _targets_adv.append(self.class_to_idx[_class_name])

            _data_adv = np.vstack(_data_adv).reshape(-1, 3, self.img_dim, self.img_dim)
            _data_adv = _data_adv.transpose((0, 2, 3, 1))  # convert to HWC
            print("_data_adv.shape:", _data_adv.shape)

            return _data_adv, _targets_adv

        if self.train == "train" or self.train == "val":
            downloaded_list = self.train_list
        elif self.train == "test":
            if adv_as_test:  # use adversarial images as the test set
                data_adv, targets_adv = _load_dav()
                self.data = data_adv
                self.targets = targets_adv
                return
            downloaded_list = self.test_list
        else:
            raise ValueError(f"error parameter train: {self.train}")

        data: Any = []
        targets = []

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                data.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])

        data = np.vstack(data).reshape(-1, self.img_dim, self.img_dim, 3)
        assert len(data) == len(targets)

        if self.train == "test":
            self.data = data
            self.targets = targets
        else:
            t2d_dict = dict({})
            for d, t in zip(data, targets):
                if t not in t2d_dict:
                    t2d_dict[t] = [d]
                else:
                    t2d_dict[t].append(d)

            data_train, data_val, targets_train, targets_val = [], [], [], []
            for t in t2d_dict.keys():  # split in each class
                cur_X = t2d_dict[t]
                if n_img > 0:  # n_img is the number of images of each class
                    cur_X = cur_X[:n_img]
                cur_y = [t for _ in range(len(cur_X))]

                X_train, X_val, y_train, y_val = train_test_split(
                    cur_X, cur_y, test_size=test_size, shuffle=True, random_state=42)

                data_train.extend(X_train)
                data_val.extend(X_val)
                targets_train.extend(y_train)
                targets_val.extend(y_val)

            self.data_train = np.asarray(data_train)
            self.data_val = np.asarray(data_val)
            self.targets_train = targets_train
            self.targets_val = targets_val

            print("self.data_train.shape:", self.data_train.shape)
            print("self.data_val.shape:", self.data_val.shape)
            print("len(self.targets_train):", len(self.targets_train))
            print("len(self.targets_val):", len(self.targets_val))

            if self.train == "train":
                self.data = self.data_train
                self.targets = self.targets_train
            else:
                self.data = self.data_val
                self.targets = self.targets_val

        # CIFAR-100 / CIFAR-10: training set (50000, 32, 32, 3), test set (10000, 32, 32, 3)
        print("self.data.shape:", self.data.shape)
        print("len(self.targets):", len(self.targets))

        # set a long-tail (imbalanced) setting for the training data
        if self.train == "train" and long_tail:
            # get a long-tail subset
            data_lt = []
            targets_lt = []
            t2d_dict = dict({})
            for d, t in zip(self.data, self.targets):
                if t not in t2d_dict:
                    t2d_dict[t] = [d]
                else:
                    t2d_dict[t].append(d)

            t_list = list(set(self.targets))
            t_list.sort()
            assert len(t_list) > 0
            ratio_gap = float(1 / len(t_list))
            cur_ratio = float(1.0)

            for t in t_list:
                d_list = t2d_dict[t]
                cur_len = int(len(d_list) * cur_ratio)
                if cur_len <= 0:
                    break
                d_list = d_list[:cur_len]
                data_lt.extend(d_list)
                targets_lt.extend([t for _ in range(len(d_list))])

                cur_ratio -= ratio_gap

            data_lt = np.vstack(data_lt).reshape(-1, self.img_dim, self.img_dim, 3)

            # CIFAR-100 / CIFAR-10: training set (50000, 32, 32, 3), test set (10000, 32, 32, 3)
            print("data_lt.shape:", data_lt.shape)
            print("len(targets_lt):", len(targets_lt))

            self.data_train = data_lt
            self.targets_train = targets_lt

        # add adversarial attacking images to the training set
        if self.train == "train" and adversarial:
            data_adv, targets_adv = _load_dav()

            self.data = np.concatenate((self.data, data_adv), axis=0)
            self.targets.extend(targets_adv)
            print("concatenated self.data.shape:", self.data.shape)

        # add synthetic images to the training set
        if self.train == "train" and n_img_syn != 0:
            data_syn: Any = []
            targets_syn = []

            root_syn = os.path.join(self.root, f"syn_data/{syn_type}/{syn_data}/")
            assert os.path.isdir(root_syn)

            class_name_list = os.listdir(root_syn)
            # class_name_list = [cn.replace("'", "") for cn in class_name_list]
            class_name_list.sort()
            for class_name in class_name_list:
                # assert label_name in self.class_to_idx
                if class_name not in self.class_to_idx:
                    continue
                img_dir = os.path.join(root_syn, class_name)
                assert os.path.isdir(img_dir)
                filename_list = os.listdir(img_dir)
                # filename_list = [int(fn[:-4]) for fn in filename_list]  # replace(".png", "")
                filename_list = [int(fn[:-4]) for fn in filename_list if fn[:-4].isdigit()]  # replace(".png", "")
                filename_list.sort()
                if n_img_syn > 0:
                    filename_list = filename_list[:n_img_syn]
                for filename in filename_list:
                    file_path = os.path.join(img_dir, f"{filename}.png")
                    assert os.path.isfile(file_path)
                    img = Image.open(file_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != self.img_size:
                        img = img.resize(size=self.img_size)
                    data_syn.append(np.array(img))
                    targets_syn.append(self.class_to_idx[class_name])

            data_syn = np.vstack(data_syn).reshape(-1, 3, self.img_dim, self.img_dim)
            data_syn = data_syn.transpose((0, 2, 3, 1))  # convert to HWC
            print("data_syn.shape:", data_syn.shape)

            self.data = np.concatenate((self.data, data_syn), axis=0)
            self.targets.extend(targets_syn)
            print("concatenated self.data.shape:", self.data.shape)

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        # print(self.classes)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


class CocoDataset(data.Dataset):

    def __init__(self, transform, mode, batch_size, vocab_threshold, pad_word, bos_word,
                 eos_word, unk_word, annotations_file, vocab_from_file, vocab_file_path, img_folder,
                 n_img: int = -1,  # the number of images per class. -1: all
                 n_img_syn: int = 0,  # the number of synthetic images per class. -1: all. 0: none.
                 ):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold=vocab_threshold,
                                pad_word=pad_word, bos_word=bos_word, eos_word=eos_word, unk_word=unk_word,
                                annotations_file=annotations_file,
                                vocab_from_file=vocab_from_file, vocab_file_path=vocab_file_path)
        self.img_folder = img_folder

        # n_img = 10000
        if self.mode == "train" or self.mode == "val":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            if n_img > 0:  # get first n_img images of all images
                self.ids = self.ids[:n_img]
            print("Obtaining caption lengths...")
            # self.caption_all_tokens = [nltk.tokenize.word_tokenize(
            #     str(self.coco.anns[self.ids[index]]["caption"]).lower())
            #     for index in tqdm(np.arange(len(self.ids)))]
            # self.caption_lengths = [len(token) for token in self.caption_all_tokens]
            # caption_all_tokens = [nltk.tokenize.word_tokenize(
            #     str(self.coco.anns[self.ids[index]]["caption"]).lower())
            #     for index in tqdm(np.arange(len(self.ids)))]
            caption_all_tokens = [nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.ids[index]]["caption"]).lower())
                for index in np.arange(len(self.ids))]
            self.caption_lengths = [len(token) for token in caption_all_tokens]
            print("len(self.caption_lengths):", len(self.caption_lengths))
        else:  # test mode
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        # Obtain image and caption if in training or validation mode
        if self.mode == "train" or self.mode == "val":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)  # shape: (3, 224, 224)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = [self.vocab(self.vocab.bos_word)]  # the caption sentence starts from the special token <bos>
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.eos_word))  # the caption sentence ends with the special token <bos>
            caption = torch.Tensor(caption).long()  # shape: (length_of_caption)

            # Return pre-processed image and caption tensors
            return image, caption, img_id

        # Obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # Return original image and pre-processed image tensor
            return orig_image, image

    def get_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where(
            [self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == "train" or self.mode == "val":
            return len(self.ids)
        else:
            return len(self.paths)


class Vocabulary:

    def __init__(self,
                 vocab_threshold=4,
                 pad_word="<pad>",
                 bos_word="<bos>",
                 eos_word="<eos>",
                 unk_word="<unk>",
                 annotations_file="../data/COCO_2015_Captioning/annotations/captions_train2014.json",
                 vocab_from_file=True,
                 vocab_file_path="../data/COCO_2015_Captioning/vocab.pkl"):
        """Initialize the vocabulary.
        Parameters:
            vocab_threshold: Minimum word count threshold.
            pad_word: Special word denoting padding words.
            bos_word: Special word denoting a sentence starts.
            eos_word: Special word denoting a sentences ends.
            unk_word: Special word denoting unknown words.
            annotations_file: Path for train annotation file.
            vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                If True, load vocab from existing vocab_file, if it exists.
            vocab_file_path: File containing the vocabulary.
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file_path = vocab_file_path
        self.pad_word = pad_word
        self.bos_word = bos_word
        self.eos_word = eos_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file

        # Load the vocabulary from file or build it from scratch.
        if self.vocab_from_file and os.path.isfile(self.vocab_file_path):
            with open(self.vocab_file_path, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print(f"Vocabulary successfully loaded from {self.vocab_file_path}")
        else:
            nltk.download("punkt")

            # Populate and initialize the dictionaries for converting tokens to integers (and vice-versa).
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0

            self.add_word(self.pad_word)  # 0
            self.add_word(self.bos_word)  # 1
            self.add_word(self.eos_word)  # 2
            self.add_word(self.unk_word)  # 3

            # Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold.
            coco = COCO(self.annotations_file)
            counter = Counter()
            ids = coco.anns.keys()
            for i, cur_id in enumerate(ids):
                caption = str(coco.anns[cur_id]["caption"])
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)

                if i % 100000 == 0:
                    print("[%d/%d] Tokenizing captions..." % (i, len(ids)))  # [0/414113]

            words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]
            for word in words:
                self.add_word(word)

            with open(self.vocab_file_path, "wb") as f:
                pickle.dump(self, f)

    def add_word(self, word):
        """Add a (special) token to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class CocoCaptionForGptFinetune(Dataset):

    def __init__(self, data) -> None:
        self.data = data

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class CocoCaptionNerForGptFinetune(Dataset):

    def __init__(self, data_cap, data_ner) -> None:
        assert isinstance(data_cap, list) and isinstance(data_ner, list) and len(data_cap) == len(data_ner) > 0
        self.data = list(zip(data_cap, data_ner))

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class CrossDomainSrcTgt(VisionDataset):
    """Office-31, Office-Hone, and ImageClef-Da-2014 cross-domain datasets.
    """

    def __init__(
            self,
            data: str = "office_31",  # "office_31", "office_hone", "imageclef_da"
            src_domain: str = "amazon",  # ["amazon", "dslr", "webcam"] ["Art", "Clipart", "Product", "RealWorld"]
            tgt_domain: str = "dslr",  # ["caltech", "imagenet", "pascal"]
            root: str = "../data/cross_domain/office_31/",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            n_img: int = -1,  # the number of images per class. -1: all
            n_img_syn: int = 0,  # the number of synthetic images per class. -1: all. 0: none.
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.syn_data = f"../data/glide_image_{data}/"
        assert os.path.isdir(self.syn_data)
        assert os.path.isdir(root)

        self.src_domain = src_domain
        self.tgt_domain = tgt_domain
        print(f"data: {data}; src_domain: {src_domain}; tgt_domain: {tgt_domain}")

        # self.data: Any = []
        # self.targets = []
        # self.image_filepath = []

        domain_dir = os.path.join(root, src_domain)
        assert os.path.isdir(domain_dir)
        class_name_list = os.listdir(domain_dir)
        class_name_list.sort()
        self.class_to_idx = dict({})
        for idx, class_name in enumerate(class_name_list):
            self.class_to_idx[class_name] = idx
        print(self.class_to_idx)

        # assert self.transform is not None
        # img_dim = 300
        # img_dim = 256
        img_dim = 224
        # img_dim = 64
        img_size = (img_dim, img_dim)  # RGB channels = 3

        def __load_domain_images(domain_name):
            domain_dir = os.path.join(root, domain_name)
            data = []
            targets = []
            image_filepath = []

            for class_name in class_name_list:
                cur_data_list = []
                cur_target_list = []
                cur_image_filepath = []

                img_dir = os.path.join(domain_dir, class_name)
                assert os.path.isdir(img_dir)
                filename_list = os.listdir(img_dir)
                if n_img > 0:  # get n_img images per class
                    filename_list = filename_list[:n_img]

                for filename in filename_list:
                    file_path = os.path.join(img_dir, filename)
                    assert os.path.isfile(file_path)
                    cur_image_filepath.append(file_path)

                    img = Image.open(file_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != img_size:  # some images are of size (img_dim, img_dim, 3)
                        img = img.resize(size=img_size)
                    # img.save("./outputs/test_1.png")

                    img_np = np.array(img)
                    cur_data_list.append(img_np)
                    # assert self.transform is not None
                    # img_tensor = self.transform(img)
                    # cur_data_list.append(img_tensor)

                    # transform_back = transforms.Compose([transforms.ToPILImage(), ])
                    # img = transform_back(img_tensor)
                    # img.save("./outputs/test_2.png")

                    target = self.class_to_idx[class_name]
                    # if self.target_transform is not None:
                    #     target = self.target_transform(target)
                    cur_target_list.append(target)

                # if n_img > 0:
                #     cur_data_list = cur_data_list[:n_img]
                #     cur_target_list = cur_target_list[:n_img]
                #     cur_image_filepath = cur_image_filepath[:n_img]

                data.extend(cur_data_list)
                targets.extend(cur_target_list)
                image_filepath.extend(cur_image_filepath)

            data = np.vstack(data).reshape(-1, 3, img_dim, img_dim)  # [N, C, H, W]
            data = data.transpose((0, 2, 3, 1))  # convert to [N, H, W, C]
            return data, targets, image_filepath

        self.data_src, self.targets_src, self.image_filepath_src = __load_domain_images(src_domain)
        self.data_tgt, self.targets_tgt, self.image_filepath_tgt = __load_domain_images(tgt_domain)

        print("self.data_src.shape:", self.data_src.shape)
        print("len(self.targets_src):", len(self.targets_src))

        print("self.data_tgt.shape:", self.data_tgt.shape)
        print("len(self.targets_tgt):", len(self.targets_tgt))

        # load the images and convert them into numpy arrays
        self.data_syn: Any = []
        self.targets_syn = []
        if n_img_syn != 0:
            root_syn = os.path.join(self.syn_data)
            assert os.path.isdir(root_syn)
            class_name_list = os.listdir(root_syn)
            # class_name_list = [cn.replace("'", "") for cn in class_name_list]
            class_name_list.sort()
            for class_name in class_name_list:
                # assert class_name in self.class_to_idx
                if class_name not in self.class_to_idx:
                    continue
                img_dir = os.path.join(root_syn, class_name)
                assert os.path.isdir(img_dir)
                filename_list = os.listdir(img_dir)
                # filename_list = [int(fn[:-4]) for fn in filename_list]  # replace(".png", "")
                filename_list = [int(fn[:-4]) for fn in filename_list if fn[:-4].isdigit()]  # replace(".png", "")
                filename_list.sort()
                if n_img_syn > 0:  # get n_img_syn images per class
                    filename_list = filename_list[:n_img_syn]
                for filename in filename_list:
                    file_path = os.path.join(img_dir, f"{filename}.png")
                    assert os.path.isfile(file_path)
                    img = Image.open(file_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != img_size:
                        img = img.resize(size=img_size)
                    self.data_syn.append(np.array(img))
                    self.targets_syn.append(self.class_to_idx[class_name])

            self.data_syn = np.vstack(self.data_syn).reshape(-1, 3, img_dim, img_dim)
            self.data_syn = self.data_syn.transpose((0, 2, 3, 1))  # convert to HWC

            print("self.data_syn.shape:", self.data_syn.shape)
            print("len(self.targets_syn):", len(self.targets_syn))

            # self.data = np.concatenate((self.data_src, self.data_tgt, self.data_syn), axis=0)
            # self.targets = self.targets_src + self.targets_tgt + self.targets_syn
            self.data = np.concatenate((self.data_src, self.data_syn), axis=0)
            self.targets = self.targets_src + self.targets_syn
        else:
            self.data = self.data_src
            self.targets = self.targets_src

        # print("Concatenated self.data.shape:", self.data.shape)
        # print("len(self.targets):", len(self.targets))

        # self.index_interval_src = (0, len(self.data_src))
        # self.index_interval_tgt = (len(self.data_src), len(self.data_src) + len(self.data_tgt))
        # self.index_interval_syn = (len(self.data_src) + len(self.data_tgt),
        #                            len(self.data_src) + len(self.data_tgt) + len(self.data_syn))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, image_tgt) where target is index of the target class.
        """
        # if self.index_interval_src[0] <= index < self.index_interval_src[1]:
        #     data = self.data_src
        #     targets = self.targets_src
        # elif self.index_interval_tgt[0] <= index < self.index_interval_tgt[1]:
        #     data = self.data_tgt
        #     targets = self.targets_tgt
        # elif self.index_interval_syn[0] <= index < self.index_interval_syn[1]:
        #     data = self.data_syn
        #     targets = self.targets_syn
        # else:
        #     raise ValueError(f"out of index: {index}")

        img, target = self.data[index], self.targets[index]
        # print(self.image_filepath[index])

        # randomly choose an image from the target domain
        img_tgt_index = random.randint(0, len(self.data_tgt) - 1)
        img_tgt = self.data_tgt[img_tgt_index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        img_tgt = Image.fromarray(img_tgt)

        if self.transform is not None:
            img = self.transform(img)
            img_tgt = self.transform(img_tgt)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img_tgt

    def __len__(self) -> int:
        return len(self.data)
        # return len(self.data_src) + len(self.data_tgt) + len(self.data_syn)


class Office31SrcTgt(VisionDataset):
    """`Office-31 <https://paperswithcode.com/sota/domain-adaptation-on-office-31>`_ Dataset.
    """

    def __init__(
            self,
            src_domain: str,  # "amazon", "dslr", "webcam"
            tgt_domain: str,  # "amazon", "dslr", "webcam"
            root: str = "../data/cross_domain/office_31/",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            n_img: int = -1,  # the number of images per class. -1: all
            n_img_syn: int = 0,  # the number of synthetic images per class. -1: all. 0: none.
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.syn_data = f"../data/glide_image_office_31/"
        self.src_domain = src_domain
        self.tgt_domain = tgt_domain

        # self.data: Any = []
        # self.targets = []
        # self.image_filepath = []

        domain_dir = os.path.join(root, src_domain)
        assert os.path.isdir(domain_dir)
        class_name_list = os.listdir(domain_dir)
        class_name_list.sort()
        self.class_to_idx = dict({})
        for idx, class_name in enumerate(class_name_list):
            self.class_to_idx[class_name] = idx
        print(self.class_to_idx)

        # assert self.transform is not None
        # img_dim = 300
        # img_dim = 256
        img_dim = 224
        # img_dim = 64
        img_size = (img_dim, img_dim)  # RGB channels = 3

        def __load_domain_images(domain_name):
            domain_dir = os.path.join(root, domain_name)
            data = []
            targets = []
            image_filepath = []

            for class_name in class_name_list:
                cur_data_list = []
                cur_target_list = []
                cur_image_filepath = []

                img_dir = os.path.join(domain_dir, class_name)
                assert os.path.isdir(img_dir)
                filename_list = os.listdir(img_dir)
                if n_img > 0:  # get n_img images per class
                    filename_list = filename_list[:n_img]

                for filename in filename_list:
                    file_path = os.path.join(img_dir, filename)
                    assert os.path.isfile(file_path)
                    cur_image_filepath.append(file_path)

                    img = Image.open(file_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != img_size:  # some images are of size (img_dim, img_dim, 3)
                        img = img.resize(size=img_size)
                    # img.save("./outputs/test_1.png")

                    img_np = np.array(img)
                    cur_data_list.append(img_np)
                    # assert self.transform is not None
                    # img_tensor = self.transform(img)
                    # cur_data_list.append(img_tensor)

                    # transform_back = transforms.Compose([transforms.ToPILImage(), ])
                    # img = transform_back(img_tensor)
                    # img.save("./outputs/test_2.png")

                    target = self.class_to_idx[class_name]
                    # if self.target_transform is not None:
                    #     target = self.target_transform(target)
                    cur_target_list.append(target)

                # if n_img > 0:
                #     cur_data_list = cur_data_list[:n_img]
                #     cur_target_list = cur_target_list[:n_img]
                #     cur_image_filepath = cur_image_filepath[:n_img]

                data.extend(cur_data_list)
                targets.extend(cur_target_list)
                image_filepath.extend(cur_image_filepath)

            data = np.vstack(data).reshape(-1, 3, img_dim, img_dim)  # [N, C, H, W]
            data = data.transpose((0, 2, 3, 1))  # convert to [N, H, W, C]
            return data, targets, image_filepath

        self.data_src, self.targets_src, self.image_filepath_src = __load_domain_images(src_domain)
        self.data_tgt, self.targets_tgt, self.image_filepath_tgt = __load_domain_images(tgt_domain)

        print("self.data_src.shape:", self.data_src.shape)
        print("len(self.targets_src):", len(self.targets_src))

        print("self.data_tgt.shape:", self.data_tgt.shape)
        print("len(self.targets_tgt):", len(self.targets_tgt))

        # load the images and convert them into numpy arrays
        self.data_syn: Any = []
        self.targets_syn = []
        if n_img_syn != 0:
            root_syn = os.path.join(self.syn_data)
            assert os.path.isdir(root_syn)
            class_name_list = os.listdir(root_syn)
            # class_name_list = [cn.replace("'", "") for cn in class_name_list]
            class_name_list.sort()
            for class_name in class_name_list:
                # assert class_name in self.class_to_idx
                if class_name not in self.class_to_idx:
                    continue
                img_dir = os.path.join(root_syn, class_name)
                assert os.path.isdir(img_dir)
                filename_list = os.listdir(img_dir)
                # filename_list = [int(fn[:-4]) for fn in filename_list]  # replace(".png", "")
                filename_list = [int(fn[:-4]) for fn in filename_list if fn[:-4].isdigit()]  # replace(".png", "")
                filename_list.sort()
                if n_img_syn > 0:
                    filename_list = filename_list[:n_img_syn]
                for filename in filename_list:
                    file_path = os.path.join(img_dir, f"{filename}.png")
                    assert os.path.isfile(file_path)
                    img = Image.open(file_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != img_size:
                        img = img.resize(size=img_size)
                    self.data_syn.append(np.array(img))
                    self.targets_syn.append(self.class_to_idx[class_name])

            self.data_syn = np.vstack(self.data_syn).reshape(-1, 3, img_dim, img_dim)
            self.data_syn = self.data_syn.transpose((0, 2, 3, 1))  # convert to HWC

            print("self.data_syn.shape:", self.data_syn.shape)
            print("len(self.targets_syn):", len(self.targets_syn))

            # self.data = np.concatenate((self.data_src, self.data_tgt, self.data_syn), axis=0)
            # self.targets = self.targets_src + self.targets_tgt + self.targets_syn
            self.data = np.concatenate((self.data_src, self.data_syn), axis=0)
            self.targets = self.targets_src + self.targets_syn
        else:
            self.data = self.data_src
            self.targets = self.targets_src

        # print("Concatenated self.data.shape:", self.data.shape)
        # print("len(self.targets):", len(self.targets))

        # self.index_interval_src = (0, len(self.data_src))
        # self.index_interval_tgt = (len(self.data_src), len(self.data_src) + len(self.data_tgt))
        # self.index_interval_syn = (len(self.data_src) + len(self.data_tgt),
        #                            len(self.data_src) + len(self.data_tgt) + len(self.data_syn))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, image_tgt) where target is index of the target class.
        """
        # if self.index_interval_src[0] <= index < self.index_interval_src[1]:
        #     data = self.data_src
        #     targets = self.targets_src
        # elif self.index_interval_tgt[0] <= index < self.index_interval_tgt[1]:
        #     data = self.data_tgt
        #     targets = self.targets_tgt
        # elif self.index_interval_syn[0] <= index < self.index_interval_syn[1]:
        #     data = self.data_syn
        #     targets = self.targets_syn
        # else:
        #     raise ValueError(f"out of index: {index}")

        img, target = self.data[index], self.targets[index]
        # print(self.image_filepath[index])

        # randomly choose an image from the target domain
        img_tgt_index = random.randint(0, len(self.data_tgt) - 1)
        img_tgt = self.data_tgt[img_tgt_index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        img_tgt = Image.fromarray(img_tgt)

        if self.transform is not None:
            img = self.transform(img)
            img_tgt = self.transform(img_tgt)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img_tgt

    def __len__(self) -> int:
        return len(self.data)
        # return len(self.data_src) + len(self.data_tgt) + len(self.data_syn)


class Office31(VisionDataset):
    """`Office-31 <https://paperswithcode.com/sota/domain-adaptation-on-office-31>`_ Dataset.
    """

    def __init__(
            self,
            domain: str,  # "amazon", "dslr", "webcam"
            root: str = "../data/cross_domain/office_31/",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            n_img: int = -1,  # the number of images per class. -1: all
            n_img_syn: int = 0,  # the number of synthetic images per class. -1: all. 0: none.
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.syn_data = f"../data/glide_image_office_31/"
        self.domain = domain

        self.data: Any = []
        self.targets = []
        self.image_filepath = []

        domain_dir = os.path.join(root, domain)
        assert os.path.isdir(domain_dir)
        class_name_list = os.listdir(domain_dir)
        class_name_list.sort()
        self.class_to_idx = dict({})
        for idx, class_name in enumerate(class_name_list):
            self.class_to_idx[class_name] = idx
        print(self.class_to_idx)

        # assert self.transform is not None
        # img_dim = 300
        # img_dim = 256
        img_dim = 224
        # img_dim = 64
        img_size = (img_dim, img_dim)  # RGB channels = 3
        for class_name in class_name_list:
            cur_data_list = []
            cur_target_list = []
            cur_image_filepath = []

            img_dir = os.path.join(domain_dir, class_name)
            assert os.path.isdir(img_dir)
            filename_list = os.listdir(img_dir)
            if n_img > 0:  # get n_img images per class
                filename_list = filename_list[:n_img]

            for filename in filename_list:
                file_path = os.path.join(img_dir, filename)
                assert os.path.isfile(file_path)
                cur_image_filepath.append(file_path)

                img = Image.open(file_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if img.size != img_size:  # some images are of size (img_dim, img_dim, 3)
                    img = img.resize(size=img_size)
                # img.save("./outputs/test_1.png")

                img_np = np.array(img)
                cur_data_list.append(img_np)
                # assert self.transform is not None
                # img_tensor = self.transform(img)
                # cur_data_list.append(img_tensor)

                # transform_back = transforms.Compose([transforms.ToPILImage(), ])
                # img = transform_back(img_tensor)
                # img.save("./outputs/test_2.png")

                target = self.class_to_idx[class_name]
                # if self.target_transform is not None:
                #     target = self.target_transform(target)
                cur_target_list.append(target)

            # if n_img > 0:
            #     cur_data_list = cur_data_list[:n_img]
            #     cur_target_list = cur_target_list[:n_img]
            #     cur_image_filepath = cur_image_filepath[:n_img]

            self.data.extend(cur_data_list)
            self.targets.extend(cur_target_list)
            self.image_filepath.extend(cur_image_filepath)

        # train (2817, 3, img_dim, img_dim)  test (498, 3, img_dim, img_dim)
        self.data = np.vstack(self.data).reshape(-1, 3, img_dim, img_dim)  # [N, C, H, W]
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to [N, H, W, C]
        # self.data = torch.stack(self.data).view(-1, 3, img_dim, img_dim)  # [N, C, H, W]
        # # self.data = self.data.permute((0, 2, 3, 1))  # convert to [N, H, W, C]

        print("self.data.shape:", self.data.shape)
        # print("self.data.size():", self.data.size())
        print("len(self.targets):", len(self.targets))

        # load the images and convert them into numpy arrays
        self.data_syn: Any = []
        self.targets_syn = []
        if n_img_syn != 0:
            root_syn = os.path.join(self.syn_data)
            assert os.path.isdir(root_syn)
            class_name_list = os.listdir(root_syn)
            # class_name_list = [cn.replace("'", "") for cn in class_name_list]
            class_name_list.sort()
            for class_name in class_name_list:
                # assert class_name in self.class_to_idx
                if class_name not in self.class_to_idx:
                    continue
                img_dir = os.path.join(root_syn, class_name)
                assert os.path.isdir(img_dir)
                filename_list = os.listdir(img_dir)
                # filename_list = [int(fn[:-4]) for fn in filename_list]  # replace(".png", "")
                filename_list = [int(fn[:-4]) for fn in filename_list if fn[:-4].isdigit()]  # replace(".png", "")
                filename_list.sort()
                if n_img_syn > 0:
                    filename_list = filename_list[:n_img_syn]
                for filename in filename_list:
                    file_path = os.path.join(img_dir, f"{filename}.png")
                    assert os.path.isfile(file_path)
                    img = Image.open(file_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != img_size:
                        img = img.resize(size=img_size)
                    self.data_syn.append(np.array(img))
                    self.targets_syn.append(self.class_to_idx[class_name])

            self.data_syn = np.vstack(self.data_syn).reshape(-1, 3, img_dim, img_dim)
            self.data_syn = self.data_syn.transpose((0, 2, 3, 1))  # convert to HWC
            print("self.data_syn.shape:", self.data_syn.shape)

            self.data = np.concatenate((self.data, self.data_syn), axis=0)
            self.targets.extend(self.targets_syn)
            print("concatenated self.data.shape:", self.data.shape)
            print("len(self.targets):", len(self.targets))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # print(self.image_filepath[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class OfficeHome(VisionDataset):
    """`Office-Home <https://paperswithcode.com/sota/domain-adaptation-on-office-home>`_ Dataset.
    """

    def __init__(
            self,
            domain: str,  # "Art", "Clipart", "Product", "RealWorld"
            root: str = "../data/cross_domain/office_home/",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            n_img: int = -1,  # the number of images per class. -1: all
            n_img_syn: int = 0,  # the number of synthetic images per class. -1: all. 0: none.
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.syn_data = f"../data/glide_image_office_home/"
        self.domain = domain

        self.data: Any = []
        self.targets = []
        self.image_filepath = []

        domain_dir = os.path.join(root, domain)
        assert os.path.isdir(domain_dir)
        class_name_list = os.listdir(domain_dir)
        class_name_list.sort()
        self.class_to_idx = dict({})
        for idx, class_name in enumerate(class_name_list):
            self.class_to_idx[class_name] = idx
        print(self.class_to_idx)

        # assert self.transform is not None
        # img_dim = 300
        # img_dim = 256
        img_dim = 224
        # img_dim = 64
        img_size = (img_dim, img_dim)  # RGB channels = 3
        # size_set = set()
        # size_set.add(img_size)
        for class_name in class_name_list:
            cur_data_list = []
            cur_target_list = []
            cur_image_filepath = []

            img_dir = os.path.join(domain_dir, class_name)
            assert os.path.isdir(img_dir)
            filename_list = os.listdir(img_dir)
            if n_img > 0:  # get n_img images per class
                filename_list = filename_list[:n_img]

            for filename in filename_list:
                file_path = os.path.join(img_dir, filename)
                assert os.path.isfile(file_path)
                cur_image_filepath.append(file_path)

                img = Image.open(file_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if img.size != img_size:  # some images are of size (img_dim, img_dim, 3)
                    # size_set.add(img.size)
                    img = img.resize(size=img_size)
                img_np = np.array(img)
                # shape_set.add(img_np.shape)
                cur_data_list.append(img_np)
                cur_target_list.append(self.class_to_idx[class_name])

            # if n_img > 0:
            #     cur_data_list = cur_data_list[:n_img]
            #     cur_target_list = cur_target_list[:n_img]
            #     cur_image_filepath = cur_image_filepath[:n_img]

            self.data.extend(cur_data_list)
            self.targets.extend(cur_target_list)
            self.image_filepath.extend(cur_image_filepath)

        self.data = np.vstack(self.data).reshape(-1, 3, img_dim, img_dim)  # train (2427. 3, img_dim, img_dim)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        print("self.data.shape:", self.data.shape)
        # print("self.data.size():", self.data.size())
        print("len(self.targets):", len(self.targets))

        # load the images and convert them into numpy arrays
        self.data_syn: Any = []
        self.targets_syn = []
        if n_img_syn != 0:
            root_syn = os.path.join(self.syn_data)
            assert os.path.isdir(root_syn)
            class_name_list = os.listdir(root_syn)
            # class_name_list = [cn.replace("'", "") for cn in class_name_list]
            class_name_list.sort()
            for class_name in class_name_list:
                # assert class_name in self.class_to_idx
                if class_name not in self.class_to_idx:
                    continue
                img_dir = os.path.join(root_syn, class_name)
                assert os.path.isdir(img_dir)
                filename_list = os.listdir(img_dir)
                # filename_list = [int(fn[:-4]) for fn in filename_list]  # replace(".png", "")
                filename_list = [int(fn[:-4]) for fn in filename_list if fn[:-4].isdigit()]  # replace(".png", "")
                filename_list.sort()
                if n_img_syn > 0:
                    filename_list = filename_list[:n_img_syn]
                for filename in filename_list:
                    file_path = os.path.join(img_dir, f"{filename}.png")
                    assert os.path.isfile(file_path)
                    img = Image.open(file_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != img_size:
                        img = img.resize(size=img_size)
                    self.data_syn.append(np.array(img))
                    self.targets_syn.append(self.class_to_idx[class_name])

            self.data_syn = np.vstack(self.data_syn).reshape(-1, 3, img_dim, img_dim)
            self.data_syn = self.data_syn.transpose((0, 2, 3, 1))  # convert to HWC
            print("self.data_syn.shape:", self.data_syn.shape)

            self.data = np.concatenate((self.data, self.data_syn), axis=0)
            self.targets.extend(self.targets_syn)
            print("concatenated self.data.shape:", self.data.shape)
            print("len(self.targets):", len(self.targets))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
