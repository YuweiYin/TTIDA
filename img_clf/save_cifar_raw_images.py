#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import time
import pickle
import numpy as np
from PIL import Image


def main() -> bool:
    timer_start = time.process_time()

    def save_cifar10_images():
        print("\n\n>>> save_cifar10_images()")
        raw_dir = "../data/cifar-10-batches-py/"
        save_dir = "../data/cifar10/"
        encoding = "utf-8"
        img_idx = dict({})

        def _unpickle(file):
            with open(file, "rb") as fo:
                data_dict = pickle.load(fo, encoding="bytes")
            return data_dict

        meta = _unpickle(os.path.join(raw_dir, "batches.meta"))
        # test_data = _unpickle(os.path.join(raw_dir, "test_batch"))

        for batch in range(1, 6):
            data = _unpickle(os.path.join(raw_dir, "data_batch_" + str(batch)))
            n_img = len(data[b"data"])
            print(f"n_img: {n_img}")
            for im_idx in range(n_img):
                im = data[b"data"][im_idx, :]
                im_r = im[0: 1024].reshape(32, 32)  # red channel
                im_g = im[1024: 2048].reshape(32, 32)  # green channel
                im_b = im[2048:].reshape(32, 32)  # blue channel
                im = np.dstack((im_r, im_g, im_b))

                cur_label = data[b"labels"][im_idx]  # integer
                cur_class = meta[b"label_names"][cur_label].decode(encoding)
                if cur_class not in img_idx:
                    img_idx[cur_class] = 0
                    os.makedirs(os.path.join(save_dir, cur_class), exist_ok=True)

                save_path = os.path.join(save_dir, cur_class, f"{img_idx[cur_class]}.png")
                print(save_path)
                im = Image.fromarray(im)
                im.save(save_path)

                img_idx[cur_class] += 1

        print(img_idx)

    def save_cifar100_images():
        print("\n\n>>> save_cifar100_images()")
        raw_dir = "../data/cifar-100-python/"
        save_dir = "../data/cifar100/"
        encoding = "utf-8"
        img_idx = dict({})

        def _unpickle(file):
            with open(file, "rb") as fo:
                data_dict = pickle.load(fo, encoding="bytes")
            return data_dict

        meta = _unpickle(os.path.join(raw_dir, "meta"))
        # test_data = _unpickle(os.path.join(raw_dir, "test"))

        data = _unpickle(os.path.join(raw_dir, "train"))
        n_img = len(data[b"data"])
        print(f"n_img: {n_img}")
        for im_idx in range(n_img):
            im = data[b"data"][im_idx, :]
            im_r = im[0: 1024].reshape(32, 32)  # red channel
            im_g = im[1024: 2048].reshape(32, 32)  # green channel
            im_b = im[2048:].reshape(32, 32)  # blue channel
            im = np.dstack((im_r, im_g, im_b))

            cur_label = data[b"fine_labels"][im_idx]  # integer
            cur_class = meta[b"fine_label_names"][cur_label].decode(encoding)
            if cur_class not in img_idx:
                img_idx[cur_class] = 0
                os.makedirs(os.path.join(save_dir, cur_class), exist_ok=True)

            save_path = os.path.join(save_dir, cur_class, f"{img_idx[cur_class]}.png")
            print(save_path)

            im = Image.fromarray(im)
            im.save(save_path)

            img_idx[cur_class] += 1

        print(img_idx)

    save_cifar10_images()
    save_cifar100_images()

    timer_end = time.process_time()
    print("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    return True


if __name__ == "__main__":
    sys.exit(main())
