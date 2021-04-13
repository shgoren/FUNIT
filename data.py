"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os.path
from PIL import Image
import cv2
import numpy as np
import torch

import torch.utils.data as data


def default_loader(path):
    try:
        im = cv2.cvtColor(cv2.imread(path.strip(), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
    except:
        raise (ValueError(path + " does not exist"))
    return Image.fromarray(np.uint8(im * 255))


def depth_im_loader(path):
    try:
        im = cv2.cvtColor(cv2.imread(path.strip(), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)[:, :,
             1]
    except:
        raise (ValueError(path + " does not exist"))
    if im.max() == np.inf:
        new_max_val = max(np.unique(im.astype(int))) + 1
        im = np.nan_to_num(im, posinf=new_max_val)
    return Image.fromarray(np.uint8(im * 255))


def default_filelist_reader(filelist):
    im_list = []
    with open(filelist, 'r') as rf:
        for line in rf.readlines():
            im_path = line.strip()
            im_list.append(im_path)
    return im_list


class ImageLabelFilelist(data.Dataset):
    def __init__(self,
                 root,
                 filelist,
                 transform=None,
                 filelist_reader=default_filelist_reader,
                 loader=default_loader,
                 return_paths=False, paired=True, source=False,
                 crop=False, imsize=224, crop_size=128):
        self.crop = crop
        self.root = root
        self.im_list = filelist_reader(os.path.join(filelist))
        self.transform = transform
        self.loader = loader
        self.paired = paired
        self.source = source
        self.imsize = imsize
        self.crop_size = crop_size
        self.classes = sorted(
            list(set([path.split('/')[0] for path in self.im_list])))
        self.fake_classes_idx = [i for i, fc in enumerate(self.classes) if 'real' not in fc.lower()]
        self.class_to_idx = {self.classes[i]: i for i in
                             range(len(self.classes))}
        self.imgs = [(im_path, self.class_to_idx[im_path.split('/')[0]]) for
                     im_path in self.im_list]
        self.order = None
        self.curr = None
        self.crops = None
        self.set_new_order()
        # get a list of all images of a class
        self.class2img_idx = {l: [] for l in range(len(self.classes))}
        [self.class2img_idx[l].append(idx) for idx, (_, l) in enumerate(self.imgs)]
        # build mapper for class and image name to index in self.imgs
        mapper = {}
        for idx, (path, label) in enumerate(self.imgs):
            image_name = os.path.basename(path)
            mapper[(label, image_name)] = idx
        self.imageXclass_mapper = mapper
        self.return_paths = return_paths
        print('Data loader')
        print("\tRoot: %s" % root)
        print("\tList: %s" % filelist)
        print("\tNumber of classes: %d" % (len(self.classes)))

    def __getitem__(self, index):
        if not self.paired:
            im_path, im_label = self.imgs[index]
            path = os.path.join(self.root, im_path)
            img = self.loader(path)
            label = im_label
        else:
            im_path, im_label = self.imgs[self.order[self.curr]]
            pseudo_random_crop = self.crops[self.curr]
            self.curr += 1
            if self.curr == len(self.imgs):
                self.set_new_order()
            if self.source:
                path = os.path.join(self.root, im_path)
                img = self.loader(path)
                label = im_label
            else:
                im_name = os.path.basename(im_path)
                # without im class and real class
                pairable_clss = self.fake_classes_idx[:im_label]+self.fake_classes_idx[im_label+1:]
                pair_cls = np.random.choice(pairable_clss)
                if 'real' not in im_path.lower():
                    pair_idx = self.imageXclass_mapper[(pair_cls, im_name)]
                else:
                    pair_idx = np.random.choice(self.class2img_idx[pair_cls])
                pair_path, target_label = self.imgs[pair_idx]
                path = os.path.join(self.root, pair_path)
                img = self.loader(path)
                label = target_label

        if self.transform is not None:
            img = self.transform(img)
        if self.paired and self.crop:
            h, w = pseudo_random_crop
            img = img[:, h:h+self.crop_size, w:w+self.crop_size]
        if self.return_paths:
            return img, label, path
        else:
            return img, label

    def __len__(self):
        return len(self.imgs)

    def set_new_order(self, seed=10):
        self.curr = 0
        np.random.seed(seed)
        self.order = np.random.permutation(len(self.imgs))
        if self.crop:
            self.crops = np.random.randint(0, self.imsize-self.crop_size, (len(self.imgs), 2))
