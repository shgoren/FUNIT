"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import random
import warnings

import torch
import os
import sys
import argparse
import shutil
from tensorboardX import SummaryWriter

from utils import get_config, get_train_loaders, make_result_folders
from utils import write_loss, write_html, write_1images, Timer, show
from trainer import Trainer



import torch.backends.cudnn as cudnn

# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/funit_animals.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path',
                    type=str,
                    default='.',
                    help="outputs path")
parser.add_argument('--multigpus',
                    action="store_true")
parser.add_argument('--batch_size',
                    type=int,
                    default=0)
parser.add_argument('--test_batch_size',
                    type=int,
                    default=4)
parser.add_argument("--resume",
                    action="store_true")
parser.add_argument('--gpu',
                    type=int,
                    default=-1, nargs='+')
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
# Override the batch size if specified.
if opts.batch_size != 0:
    config['batch_size'] = opts.batch_size

trainer = Trainer(config)
if len(opts.gpu) == 1 and opts.gpu[0] > -1:
    config['gpus'] = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu[0])
    trainer.cuda()
elif isinstance(opts.gpu, list) and opts.gpu:
    ngpus = torch.cuda.device_count()
    config['gpus'] = ngpus
    print("Number of GPUs: %d" % ngpus)
    trainer.cuda()
    trainer.model = torch.nn.DataParallel(
        trainer.model, device_ids=opts.gpu)
else:
    raise ValueError("specific gpu not selected")

train_loader, test_loader = get_train_loaders(config)

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = SummaryWriter(
    os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

iterations = trainer.resume(checkpoint_directory,
                            hp=config,
                            multigpus=opts.multigpus) if opts.resume else 0

# warnings.simplefilter('error')

while True:
    for it, (cont_data, style_data, pair_data) in enumerate(train_loader):
        # returned cont data dimensions is (b, c, h, w)
        # returned style data dimensions is (b, k, c, h, w)
        with Timer("Elapsed time in update: %f"):
            if it % config.get('dis_skip', 1) == 0:
                d_acc = trainer.dis_update(cont_data, style_data, config)
            g_acc = trainer.gen_update(cont_data, style_data, pair_data, config,
                                       opts.multigpus)
            torch.cuda.synchronize()
            print('D acc: %.4f\t G acc: %.4f' % (d_acc, g_acc))

        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        if ((iterations + 1) % config['image_save_iter'] == 0 or (
                iterations + 1) % config['image_display_iter'] == 0):
            if (iterations + 1) % config['image_save_iter'] == 0:
                key_str = '%08d' % (iterations + 1)
                write_html(output_directory + "/index.html", iterations + 1,
                           config['image_save_iter'], 'images')
            else:
                key_str = 'current'
            with torch.no_grad():
                for t, (val_cont_data, val_style_data, pair_data) in enumerate(train_loader):
                    if t >= opts.test_batch_size:
                        break
                    cont, recon_curr, trans_curr, style, recon, trans = trainer.test(val_cont_data, val_style_data,
                                                                                     opts.multigpus)
                    if torch.all(pair_data[0] == pair_data[0]):
                        xp = pair_data[0].cuda()
                        val_image_outputs = (cont, style[:, 0], trans, xp, recon)
                    else:
                        val_image_outputs = (cont, style[:, 0], trans, recon)
                    write_1images(val_image_outputs, image_directory,
                                  'train_%s_%02d' % (key_str, t))

                for t, (test_co_data, test_cl_data, _) in enumerate(test_loader):
                    if t >= opts.test_batch_size:
                        break
                    cont, recon_curr, trans_curr, style, recon, trans = trainer.test(test_co_data,
                                                                                     test_cl_data,
                                                                                     opts.multigpus)
                    test_image_outputs = (cont, style[:, 0], trans, recon)
                    write_1images(test_image_outputs, image_directory,
                                  'test_%s_%02d' % (key_str, t))
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations, opts.multigpus)
            print('Saved model at iteration %d' % (iterations + 1))

        iterations += 1
        if iterations >= max_iter:
            print("Finish Training")
            sys.exit(0)

