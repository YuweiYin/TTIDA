# -*- coding:utf-8 -*-

import os
from cdtrans.datasets.bases import BaseImageDataset


class DomainNet(BaseImageDataset):
    dataset_dir = ''

    def __init__(self, cfg, root_train='./datasets/reid_datasets/Corrected_Market1501',
                 root_val='./datasets/reid_datasets/Corrected_Market1501', pid_begin=0, verbose=True,
                 syn_num: int = 0, **kwargs):
        super(DomainNet, self).__init__()

        self.cfg = cfg
        self.syn_num = syn_num
        # self.syn_dir = "../data/glide_image_domainnet/"
        # assert os.path.isdir(self.syn_dir)

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
        train = self._process_dir(root_train, self.train_dataset_dir)
        # test = self._process_dir(root_test)
        valid = self._process_dir(root_val, self.valid_dataset_dir)

        if verbose:
            print(">>> DomainNet dataset loaded")
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

    def _process_dir(self, list_path, dir_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        # print(lines)
        for img_idx, img_info in enumerate(lines):
            data = img_info.split(' ')
            # print(data)
            if len(data) == 1:
                print(data)
            img_path, pid = data
            pid = int(pid)  # no need to relabel
            img_path = os.path.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin + pid, 0, 0, img_idx))
            pid_container.add(pid)
        #             cam_container.add(camid)
        #         print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        # for idx, pid in enumerate(pid_container):
        #     assert idx == pid, "See code comment for explanation"
        return dataset
