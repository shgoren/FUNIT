"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from tqdm import tqdm

from utils import get_config
from trainer import Trainer
from data import depth_im_loader

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/funit_animals.yaml')
parser.add_argument('--ckpt',
                    type=str,
                    default='pretrained/animal119_gen_00200000.pt')
parser.add_argument('--class_image_folder',
                    type=str,
                    default='datasets/apple/Real_Kinect_Depth_preprocessed')
parser.add_argument('--input',
                    type=str,
                    default='datasets/apple/1000_Depth_NotStatic_preprocessed/')
parser.add_argument('--output',
                    type=str,
                    default='images/12_after_unpaired')
parser.add_argument('--gpu',
                    type=int,
                    default=-1, nargs='+')
opts = parser.parse_args()
if not os.path.exists(opts.output):
    os.mkdir(opts.output)

cudnn.benchmark = True
opts.vis = True
config = get_config(opts.config)
config['batch_size'] = 1
config['gpus'] = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu[0])

trainer = Trainer(config)
trainer.cuda()
trainer.load_ckpt(opts.ckpt)
trainer.eval()

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))]
transform_list = [transforms.Resize((224, 224))] + transform_list
transform = transforms.Compose(transform_list)

print('Compute average class codes for images in %s' % opts.class_image_folder)
images = os.listdir(opts.class_image_folder)
for i, f in tqdm(enumerate(images)):
    fn = os.path.join(opts.class_image_folder, f)
    img = depth_im_loader(fn)
    img_tensor = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        class_code = trainer.model.compute_k_style(img_tensor, 1)
        if i == 0:
            new_class_code = class_code
            all_class_codes = class_code
        else:
            new_class_code += class_code
            all_class_codes = torch.cat([all_class_codes, class_code], 0)


final_class_code = new_class_code / len(images)
average_std = all_class_codes.std(dim=0).mean()
print(f"average entry std in class code: {average_std}")

print(f"translating")
print('Saving output to %s' % opts.output)
for im_name in tqdm(sorted(os.listdir(opts.input))):
    image = depth_im_loader(os.path.join(opts.input, im_name))
    content_img = transform(image).unsqueeze(0)

    print('Compute translation for %s' % opts.input)
    with torch.no_grad():
        output_image = trainer.model.translate_simple(content_img, final_class_code)
        image = output_image.detach().cpu().squeeze().numpy()
        # image = np.transpose(image, (1, 2, 0))
        image = ((image + 1) * 0.5 * 255.0)
        output_img = Image.fromarray(np.uint8(image))
        output_img.save(os.path.join(opts.output, im_name[:-4])+".JPEG", 'JPEG', quality=99)


