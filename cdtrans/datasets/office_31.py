# -*- coding:utf-8 -*-

import os
# from PIL import Image
from cdtrans.datasets.bases import BaseImageDataset


class Office31(BaseImageDataset):
    dataset_dir = ''

    def __init__(self, cfg, root_train='./datasets/reid_datasets/Corrected_Market1501',
                 root_val='./datasets/reid_datasets/Corrected_Market1501', pid_begin=0, verbose=True,
                 syn_num: int = 0, syn_type: str = "base", **kwargs):
        super(Office31, self).__init__()

        self.cfg = cfg
        data = cfg.DATASETS.DATA
        # source_domain = cfg.DATASETS.SOURCE_DOMAIN
        # target_domain = cfg.DATASETS.TARGET_DOMAIN
        # n_img_syn = cfg.DATASETS.SYN_NUM

        self.syn_num = syn_num

        syn_data = "office_31"
        # self.syn_dir = f"../data/glide/glide_image_base/{syn_data}/"
        # self.syn_dir = f"../data/glide/glide_image_upsample/{syn_data}/"
        self.syn_dir = f"../data/glide/glide_image_{syn_type}/{syn_data}/"
        assert os.path.isdir(self.syn_dir)

        root_train = root_train
        # root_test = root_test
        root_val = root_val
        self.train_dataset_dir = os.path.dirname(root_train)
        self.valid_dataset_dir = os.path.dirname(root_val)
        self.train_name = os.path.basename(root_train).split('.')[0]
        # self.test_name = os.path.dirname(root_test).split('/')[-1]
        self.val_name = os.path.basename(root_val).split('.')[0]
        self.test_name = self.val_name
        self.pid_begin = pid_begin

        train = self._process_dir(root_train, self.train_dataset_dir, syn_num=syn_num, syn_dir=self.syn_dir)
        valid = self._process_dir(root_val, self.valid_dataset_dir, syn_num=0, syn_dir=self.syn_dir)
        # test = self._process_dir(root_test)

        if verbose:
            print(">>> Office-31 dataset loaded")
            self.print_dataset_statistics(train, valid)

        self.train = train
        self.test = valid
        self.valid = valid

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_test_pids, self.num_test_imgs, self.num_test_cams, self.num_test_vids = self.get_imagedata_info(
            self.test)
        self.num_valid_pids, self.num_valid_imgs, self.num_valid_cams, self.num_valid_vids = self.get_imagedata_info(
            self.valid)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.art_dir):
            raise RuntimeError("'{}' is not available".format(self.art_dir))
        if not os.path.exists(self.clipart_dir):
            raise RuntimeError("'{}' is not available".format(self.clipart_dir))
        if not os.path.exists(self.product_dir):
            raise RuntimeError("'{}' is not available".format(self.product_dir))
        if not os.path.exists(self.realworld_dir):
            raise RuntimeError("'{}' is not available".format(self.realworld_dir))

    def print_dataset_statistics(self, train, test):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_test_pids, num_test_imgs, num_test_cams, num_targe_views = self.get_imagedata_info(test)
        # num_valid_pids, num_valid_imgs, num_valid_cams, num_valid_views = self.get_imagedata_info(valid)

        print("Dataset statistics:")
        print("train {} and test is {}".format(self.train_name, self.test_name))
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train   | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  test    | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_cams))
        print("  ----------------------------------------")

    def _process_dir(self, list_path, dir_path, syn_num: int = 0, syn_dir=""):
        assert os.path.isfile(list_path)
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()

        class_to_label = dict({})
        label_to_class = dict({})
        img_idx = 0
        for img_info in lines:
            data = img_info.strip().split()
            if len(data) == 1:
                print(data)
            img_path, pid = data
            pid = int(pid)  # no need to relabel
            class_name = img_path.split("/")[-2]

            if class_name not in class_to_label:
                class_to_label[class_name] = pid
                label_to_class[pid] = class_name
            else:
                assert class_to_label[class_name] == pid
                assert label_to_class[pid] == class_name

            img_path = os.path.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin + pid, 0, 0, img_idx))
            pid_container.add(pid)
            img_idx += 1

        # load the images and convert them into numpy arrays
        # img_dim = 224
        # img_size = (img_dim, img_dim)  # RGB channels = 3
        dataset_syn = []
        pid_container_syn = set()
        if syn_num != 0:
            assert os.path.isdir(syn_dir)
            class_name_list = os.listdir(syn_dir)
            # class_name_list = [cn.replace("'", "") for cn in class_name_list]
            # class_name_list.sort()
            for class_name in class_name_list:
                # assert class_name in class_to_label
                if class_name not in class_to_label:
                    continue
                img_dir = os.path.join(syn_dir, class_name)
                assert os.path.isdir(img_dir)
                filename_list = os.listdir(img_dir)
                filename_list = [int(fn[:-4]) for fn in filename_list]  # replace(".png", "")
                filename_list.sort()
                if syn_num > 0:  # get n_img_syn images per class
                    filename_list = filename_list[:syn_num]
                for filename in filename_list:
                    # file_path = os.path.join(img_dir, filename)
                    file_path = os.path.join(img_dir, f"{filename}.png")
                    assert os.path.isfile(file_path)
                    # img = Image.open(file_path)
                    # if img.mode != "RGB":
                    #     img = img.convert("RGB")
                    # if img.size != img_size:
                    #     img = img.resize(size=img_size)
                    # self.data_syn.append(np.array(img))
                    # self.targets_syn.append(self.class_to_idx[class_name])
                    pid = class_to_label[class_name]
                    dataset_syn.append((file_path, self.pid_begin + pid, 0, 0, img_idx))
                    pid_container_syn.add(pid)
                    img_idx += 1

        print(f"n_img_origin:{len(dataset)}; n_img_syn: {len(dataset_syn)}; "
              f"n_img_all: {len(dataset) + len(dataset_syn)}")
        dataset = dataset + dataset_syn
        pid_container = pid_container | pid_container_syn
        return dataset
