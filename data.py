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
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

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
            if self.crop:
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
                pairable_clss = self.fake_classes_idx[:im_label] + self.fake_classes_idx[im_label + 1:]
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
            img = img[:, h:h + self.crop_size, w:w + self.crop_size]
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
            self.crops = np.random.randint(0, self.imsize - self.crop_size, (len(self.imgs), 2))


class ContentStyleDataset(data.Dataset):

    def __init__(self,
                 root,
                 filelist,
                 transform=None,
                 filelist_reader=default_filelist_reader,
                 im_loader=default_loader,
                 paired=False,
                 patch_size=None,
                 syle_all_patches=True,
                 k=1,
                 return_path=False):
        """

        :param root: data folder.
        :param filelist: file containing all files to use for training.
                         expecting images of different classes in different folders
        :param transform: torch transforms object
        :param filelist_reader: function to read the filelist
        :param im_loader: loader function for a single image path.
        :param paired: bool, is the dataset paired?
                       if dataset is paired then the translation expected result
                       is returned together with content and style.
        :param patch_size: bool, work with patches of whole image
        :param syle_all_patches: bool, if True than style is calculaed over all image patches
                                 (used only when `patches == True`)
        :param k: how many images to use for style calculaion.
        """
        assert (not (paired and k > 1)), "cannot use k>1 together with paired mode"
        self.root = root
        self.im_list = filelist_reader(os.path.join(filelist))
        self.transform = transform
        self.im_loader = im_loader
        self.paired = paired
        self.k = k
        self.syle_all_patches = syle_all_patches
        self.patch_size = patch_size
        self.return_path = return_path

        self.classes = sorted(list(set([path.split('/')[0] for path in self.im_list])))
        self.fake_classes_idx = [i for i, fc in enumerate(self.classes) if 'real' not in fc.lower()]
        self.class2label = {self.classes[i]: i for i in range(len(self.classes))}
        self.label2class = {v: k for k, v in self.class2label.items()}
        self.imgs = {i: {} for i in range(len(self.classes))}
        for im_p in self.im_list:
            im_id, im_lbl = self._id_label_from_path(im_p)
            self.imgs[im_lbl][im_id] = im_p

    def __getitem__(self, idx):
        """
        :param idx: a tuple of (content_idx, style_idx).
        :return (content_image, style_image, expected_result [if paired] or None [if not paired] )
        """
        # content
        cont_idx, style_idx = idx
        cont_im_p = self.im_list[cont_idx]
        cont_id, cont_lbl = self._id_label_from_path(cont_im_p)
        cont_im = self.transform(self.im_loader(os.path.join(self.root, cont_im_p)))
        # style
        style_im_p = self.im_list[style_idx]
        style_id, style_lbl = self._id_label_from_path(style_im_p)
        if self.k > 1:
            # choose random k-1 images of same label
            style_id = [style_id] + np.random.choice(list(self.imgs[style_lbl].keys()), size=self.k - 1).tolist()
            style_im_p = [self.imgs[style_lbl][im_id] for im_id in style_id]
            style_lbl = [style_lbl] * self.k
        else:
            style_im_p, style_id, style_lbl = [style_im_p], [style_id], [style_lbl]
        style_im = [self.transform(self.im_loader(os.path.join(self.root, im_p))) for im_p in style_im_p]

        if self.patch_size is not None:
            cont_im = extract_patches(cont_im, self.patch_size)
            style_im = [extract_patches(im, self.patch_size) for im in style_im]
            random_patch = np.random.choice(range(len(cont_im)))
            cont_im = cont_im[random_patch]
            if self.syle_all_patches:
                patch_cnt = len(style_im[0])
                style_im = flatten(style_im)
                style_id = flatten([[im_id] * patch_cnt for im_id in style_id])
                style_lbl = flatten([[im_lbl] * patch_cnt for im_lbl in style_lbl])
            else:
                style_im = [im_patch_list[random_patch] for im_patch_list in style_im]

        if self.paired:
            res_im_p = self.imgs[style_lbl[0]][cont_id]
            res_id = cont_id
            res_lbl = style_lbl
            res_im = self.transform(self.im_loader(os.path.join(self.root, res_im_p)))
            if self.patch_size is not None:
                res_im = extract_patches(res_im, self.patch_size)[random_patch]
            result = [res_im, res_lbl]
        else:
            result = [np.nan]

        content = [cont_im, cont_lbl]
        style = [torch.stack(style_im, 0), torch.tensor(style_lbl)]

        if self.return_path:
            content.append(cont_im_p)
            style.append(style_im_p)
            if self.paired:
                result.append(res_im_p)

        return content, style, result

    def __len__(self):
        return len(self.im_list)

    def _id_label_from_path(self, im_p):
        im_name = os.path.splitext(os.path.basename(im_p))[0]
        im_cls = im_p.split('/')[0]
        im_lbl = self.class2label[im_cls]
        return im_name, im_lbl


def extract_patches(im, patch_size, overlap=0):
    if isinstance(im, Image.Image):
        im = np.array(im)
    patches = []
    c, h, w = im.shape
    for x in range(0, h, patch_size - overlap):
        for y in range(0, w, patch_size - overlap):
            if y + patch_size > w:
                y = w - patch_size
            if x + patch_size > h:
                x = h - patch_size
            patch = im[:, x:x + patch_size, y:y + patch_size]
            patches.append(patch)
    return patches


def flatten(list_of_lists):
    return [y for x in list_of_lists for y in x]

# --------------------------------------------------------

def test_dataset():
    dataset = ContentStyleDataset("./datasets/apple", "./datasets/6_train_fake+noises_2_test_real_comp__v2_train.txt",
                                  transforms.Resize(224),
                                  im_loader=depth_im_loader,
                                  paired=False,
                                  patch_size=None,
                                  return_path=True,
                                  syle_all_patches=True,
                                  k=3)
    c, s, r = dataset[0, 2008]
    print(c, s, r)
    plt.imshow(c[0], cmap='gray')
    plt.show()
    plt.imshow(s[0][0], cmap='gray')
    plt.show()
    # plt.imshow(r[0], cmap='gray')
    # plt.show()
