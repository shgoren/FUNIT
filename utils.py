"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import yaml
import time

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
import torchvision.utils as vutils
import math
from torchvision.transforms import transforms
import cv2
import numpy as np
irange = range
import torch.nn.functional as F

from data import ImageLabelFilelist, depth_im_loader


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


def loader_from_list(
        root,
        file_list,
        batch_size,
        new_size=None,
        crop_size=128,
        crop=True,
        num_workers=4,
        shuffle=True,
        center_crop=False,
        return_paths=False,
        drop_last=True,
        k=1, paired=False, hflip=True, source=False):
    """

    :param root:
    :param file_list:
    :param batch_size:
    :param new_size:
    :param crop_size:
    :param crop:
    :param num_workers:
    :param shuffle:
    :param center_crop:
    :param return_paths:
    :param drop_last:
    :param k:
    :param paired: get images in a preset order
    :param hflip:
    :param source: use only with paired, indicates which part of the pair to return
    :return:
    """
    assert not (k > 1 and paired), "can't use paired training with k>1"
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.5,))
                      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ]
    if center_crop:
        transform_list = [transforms.CenterCrop(crop_size)] + \
                         transform_list if crop else transform_list
    else:
        transform_list = [transforms.RandomCrop(crop_size)] + \
                         transform_list if crop and not paired else transform_list
    transform_list = [transforms.Resize((new_size, new_size))] + transform_list \
        if new_size is not None else transform_list
    if not center_crop and hflip:
        transform_list = [transforms.RandomHorizontalFlip()] + transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageLabelFilelist(root,
                                 file_list,
                                 transform,
                                 loader=depth_im_loader,
                                 return_paths=return_paths, crop=crop,
                                 paired=paired, source=source, imsize=new_size, crop_size=crop_size)
    loader = DataLoader(dataset,
                        num_workers=num_workers,
                        batch_sampler=BatchSamplerKImages(dataset, batch_size, k))
    return loader


class BatchSamplerKImages:

    def __init__(self, dataset, batch, k):
        self.dataset = dataset
        self.batch_size = batch
        self.k = k
        self.class2img_idx = dataset.class2img_idx
        self.sampler = RandomSampler(dataset)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            _, label = self.dataset.imgs[idx]
            for _ in range(self.k-1):
                idx = np.random.choice(self.class2img_idx[label])
                batch.append(idx)
            if len(batch) == self.batch_size * self.k:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch




def get_evaluation_loaders(conf, shuffle_content=False):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    crop_size = conf['crop_size']
    content_loader = loader_from_list(
        root=conf['data_folder_train'],
        file_list=conf['data_list_train'],
        batch_size=batch_size,
        new_size=new_size,
        crop_size=crop_size,
        crop=True,
        num_workers=num_workers,
        shuffle=shuffle_content,
        center_crop=False,
        return_paths=True,
        drop_last=False, paired=False, source=True)

    class_loader = loader_from_list(
        root=conf['data_folder_test'],
        file_list=conf['data_list_test'],
        batch_size=batch_size * conf['k_shot'],
        new_size=new_size,
        crop_size=crop_size,
        crop=True,
        num_workers=1,
        shuffle=False,
        center_crop=False,
        return_paths=True,
        drop_last=False, paired=False, source=False)
    return content_loader, class_loader

def get_train_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    crop_size = conf['crop_size']
    train_content_loader = loader_from_list(
        root=conf['data_folder_train'],
        file_list=conf['data_list_train'],
        batch_size=batch_size,
        new_size=new_size,
        crop_size=crop_size,
        crop=conf.get('crop', False),
        return_paths=True,
        num_workers=num_workers, paired=conf['paired'], hflip=False, source=True)
    train_class_loader = loader_from_list(
        root=conf['data_folder_train'],
        file_list=conf['data_list_train'],
        batch_size=batch_size,
        new_size=new_size,
        crop_size=crop_size,
        crop=conf.get('crop', False),
        return_paths=True,
        num_workers=num_workers,
        k=conf['k_class_images'], paired=False, hflip=False)
    if conf['paired']:
        paired_content_loader = loader_from_list(
            root=conf['data_folder_train'],
            file_list=conf['data_list_train'],
            batch_size=batch_size,
            new_size=new_size,
            crop_size=crop_size,
            crop=conf.get('crop', False),
            return_paths=True,
            num_workers=num_workers, paired=True, hflip=False, source=False)
    else:
        paired_content_loader = [None] * len(train_content_loader.dataset)



    test_content_loader = loader_from_list(
        root=conf['data_folder_test'],
        file_list=conf['data_list_test'],
        batch_size=batch_size,
        new_size=new_size,
        crop_size=crop_size,
        crop=False,
        return_paths=True,
        num_workers=0, paired=False, hflip=False, source=True)
    test_class_loader = loader_from_list(
        root=conf['data_folder_test'],
        file_list=conf['data_list_test'],
        batch_size=batch_size,
        new_size=new_size,
        crop_size=crop_size,
        crop=False,
        return_paths=True,
        num_workers=0,
        k=conf['k_class_images'], paired=False, hflip=False, source=False)

    return (train_content_loader, train_class_loader, paired_content_loader,
            test_content_loader, test_class_loader)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def make_result_folders(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory



def __write_images(im_outs, dis_img_n, file_name):
    # im_outs = [images.expand(-1, 3, -1, -1) for images in im_outs]
    im_outs = [images.expand(-1, 1, -1, -1) for images in im_outs]
    image_tensor = torch.cat([F.interpolate(images[:dis_img_n], size=224) for images in im_outs], 0)
    image_grid = make_grid_with_labels(image_tensor.data,
                                       ["content", "current reconstruction", "current translation", "class",
                                        "aggregate reconstruction", "aggregate translation"],
                                       nrow=dis_img_n, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def make_grid_with_labels(tensor, labels, nrow=8, limit=1000, padding=2,
                          normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        labels (list):  ( [labels_1,labels_2,labels_3,...labels_n]) where labels is Bx1 vector of some labels
        limit ( int, optional): Limits number of images and labels to make grid of
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    # Opencv configs
    if not isinstance(labels, list):
        raise ValueError
    else:
        labels = np.asarray(labels)
    if limit is not None:
        tensor = tensor[:limit, ::]
        labels = labels[:limit]

    font = 1
    fontScale = 2
    color = (255, 0, 0)
    thickness = 1

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            working_tensor = tensor[k]
            if labels is not None:
                org = (0, int(tensor[k].shape[1] * 0.1))
                working_image = cv2.UMat(
                    np.asarray(np.transpose(working_tensor.cpu().numpy(), (1, 2, 0)) * 255).astype('uint8'))
                if x == 0:
                    working_image = cv2.putText(working_image, f'{str(labels[y])}', org, font,
                                        fontScale, color, thickness, cv2.LINE_AA)
                working_tensor = transforms.ToTensor()(working_image.get())

            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(working_tensor)
            k = k + 1
    return grid


def write_1images(image_outputs, image_directory, postfix):
    display_image_num = image_outputs[0].size(0)
    __write_images(image_outputs, display_image_num,
                   '%s/gen_%s.jpg' % (image_directory, postfix))


def _write_row(html_file, it, fn, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (it, fn.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (fn, fn, all_size))
    return


def write_html(filename, it, img_save_it, img_dir, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    _write_row(html_file, it, '%s/gen_train_current.jpg' % img_dir, all_size)
    for j in range(it, img_save_it - 1, -1):
        _write_row(html_file, j, '%s/gen_train_%08d.jpg' % (img_dir, j),
                   all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
