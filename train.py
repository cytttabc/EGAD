import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_msssim import ssim, SSIM
from skimage.metrics import peak_signal_noise_ratio
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import time
import shutil
import torch.backends.cudnn as cudnn
from utils import AverageMeter,ssim,psnr
from datasets.loader import PairLoader
from utils.utils import create_logger, summary_model, \
    save_checkpoint, resume_checkpoint, save_model, \
    set_seed_torch
from torchvision.models import vgg16
from models import *
from models.nets4 import Dehaze,Get_gradient_nopadding
import warnings

# 在你的代码开始处添加以下代码，以忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='nets4-t', type=str, help='model')
parser.add_argument('--model_name', default='nets4.py', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='result4', type=str,
                    help='path to models saving')
parser.add_argument('--resume_checkpoint', default='', type=bool,
                    help='resume checkpoint')

# dataset config
parser.add_argument('--datasets_dir', default='new_hazydata3_resize', type=str, help='path to datasets dir')
parser.add_argument('--train_dataset', default='train640480', type=str, help='train dataset name')
parser.add_argument('--valid_dataset', default='val640480', type=str, help='valid dataset name')
parser.add_argument('--exp_config', default='outdoor', type=str, help='experiment configuration')
parser.add_argument('--exp_name', default='test', type=str, help='experiment name')



parser.add_argument('--cudnn_BENCHMARK', default=True)
parser.add_argument('--cudnn_DETERMINISTIC', default=False)
parser.add_argument('--cudnn_ENABLED', default=True)

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
get_gradient_model = Get_gradient_nopadding().to(device)
def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    torch.cuda.empty_cache()

    network.train()
    pbar = tqdm(desc="Epoch[{0}]".format(epoch), total=len(train_loader), leave=True,
                ncols=160)
    end = time.time()
    for batch in train_loader:
        source_img = batch['source'].to(device)
        target_img = batch['target'].to(device)



        # loss = loss1 + 0.04 * loss2
        # loss=0.5*loss1+0.3*loss2+0.1*loss3+0.1*loss4
        # loss=0.5*loss1+0.2*loss2+0.1*loss3+0.2*loss5

        with autocast(args.no_autocast):
            out,feature= network(source_img)
            # out=out.to(device)
            # feature = feature.to(device)
            gradient = get_gradient_model(target_img)
            loss1 = criterion[0](out, target_img)
            loss2 = criterion[1](out, target_img)
            loss3 = criterion[0](feature, gradient)
            # loss4= criterion[1](feature, gradient)
            gradient_out = get_gradient_model(out)
            loss5 = criterion[0](gradient_out, gradient)
            loss = loss1 + 0.01*loss2 + 0.01 * loss3 + 0.01 * loss5

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(Speed="{:.1f} samples/s".format(out.size(0) / batch_time.val),
                         Loss="{:.5f}".format(loss))
        pbar.update()
    pbar.close()
    return losses.avg
# def valid(val_loader, network):
#     losses = AverageMeter()
#     PSNR = AverageMeter()
#     SSIM = AverageMeter()
#
#     torch.cuda.empty_cache()
#
#     network.eval()
#     pbar = tqdm(desc="Testing", total=len(val_loader), leave=True, ncols=160)
#     for batch in val_loader:
#         source_img = batch['source'].to(device)
#         target_img = batch['target'].to(device)
#
#         with torch.no_grad():
#             output, feature = network(source_img)
#
#             output = output * 0.5 + 0.5
#             target_img = target_img * 0.5 + 0.5
#
#             mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
#             psnr_mimo = 10 * torch.log10(1 / mse_loss).mean()
#
#             _, _, H, W = output.size()
#             down_ratio = max(1, round(min(H, W) / 256))
#             ssim1_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
#                              F.adaptive_avg_pool2d(target_img, (int(H / down_ratio), int(W / down_ratio))),
#                              data_range=1, size_average=False)
#             loss1 = criterion[0](output, target_img)
#             loss = loss1
#
#         losses.update(loss.item())
#         pbar.set_postfix(PSNR="{:.2f}db".format(psnr_mimo))
#         pbar.update()
#         # print(ssim1_val)
#         PSNR.update(psnr_mimo.item(), source_img.size(0))
#         SSIM.update(ssim1_val.item(), source_img.size(0))
#     pbar.close()
#     return losses.avg, PSNR.avg, SSIM.avg

def valid(val_loader, network):
    losses = AverageMeter()
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    torch.cuda.empty_cache()

    network.eval()
    # end = time.time()
    # init progress bar

    # pbar = tqdm(desc="Testing", total=len(val_loader), leave=True, ncols=160)
    for batch in val_loader:
        source_img = batch['source'].to(device)
        target_img = batch['target'].to(device)

        with torch.no_grad():  # torch.no_grad() may cause warning
            output,feature = network(source_img)
            loss1 = criterion[0](output, target_img)
            # loss2 = criterion[1](output, target_img)
            loss = loss1

        losses.update(loss.item())

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        # mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        # pbar.set_postfix(PSNR="{:.2f}db".format(psnr))
        # pbar.update()
        ssim1 = ssim(output, target_img)
        SSIM.update(ssim1.item(), source_img.size(0))
        PSNR.update(psnr.item(), source_img.size(0))
    # pbar.close()
    return losses.avg, PSNR.avg,SSIM.avg

def setup_cudnn(config):
    cudnn.benchmark = config.cudnn_BENCHMARK
    torch.backends.cudnn.deterministic = config.cudnn_DETERMINISTIC
    torch.backends.cudnn.enabled = config.cudnn_ENABLED


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp_config, args.model + '.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    # set random seed
    set_seed_torch()

    setup_cudnn(args)

    # Create logger
    final_output_dir = os.path.join(args.save_dir, args.train_dataset, args.exp_name)
    create_logger(final_output_dir)
    # network = torch.nn.DataParallel(Dehaze()).cuda()
    # device=torch.device('cuda:2')
    # network=network.to(device)
    network=Dehaze().to(device)
    # build network
    # network = model.eval().to(device)

    # copy config file
    shutil.copy2(
        setting_filename,
        final_output_dir
    )

    # copy model file
    summary_model(network, args.model_name, final_output_dir, [256, 256])

    # network = torch.nn.DataParallel(network)
    # network.to(device)
    # build criterion
    criterion = []
    criterion.append(nn.L1Loss().to(device))

    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterion.append(PerLoss(vgg_model).to(device))

    # build optimizer
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    # resume checkpoint
    best_psnr, begin_epoch = resume_checkpoint(network, optimizer, args, final_output_dir, True)

    # build scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2, last_epoch=begin_epoch - 1)
    # build scaler
    scaler = GradScaler()

    # build dataloader
    train_dataset = PairLoader(args.datasets_dir, args.train_dataset, 'train',
                               setting['patch_size'], setting['only_h_flip'])
    val_dataset = PairLoader(args.datasets_dir,args.valid_dataset, 'valid',
                                setting['valid_mode'])

    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=16,
                            num_workers=args.num_workers,
                            pin_memory=True)
    # init SummaryWriter

    with SummaryWriter(log_dir=final_output_dir) as writer:
    # begin epoch
     logging.info('=> start training')

     for epoch in range(begin_epoch, setting['epochs'] + 1):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} train start'.format(head))
        lr = scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        start = time.time()
        train_loss = train(train_loader, network, criterion, optimizer, scaler)
        writer.add_scalars('Loss', {'train Loss': train_loss}, epoch)
        msg = '=> Train:\t' \
              'Loss {:.4f}\t'.format(train_loss)
        logging.info(msg)
        logging.info('=> {} train end, duration: {:.2f}s'.format(head, time.time() - start))

        scheduler.step(epoch=epoch + 1)

        save_checkpoint(model=network, model_name=network, optimizer=optimizer,
                        output_dir=final_output_dir, in_epoch=True, epoch_or_step=epoch, best_perf=best_psnr)

        if epoch % setting['eval_freq'] == 0:
            logging.info('=> {} validate start'.format(head))

            val_start = time.time()
            valid_loss, avg_psnr,avg_ssim = valid(val_loader, network)
            writer.add_scalars('Loss', {'valid Loss': valid_loss}, epoch)
            msg = '=> Valid:\t' \
                  'Loss {:.4f}\t' \
                  'PSNR {:.2f}\t' \
                  'SSIM {:.2f}\t' .format(valid_loss, avg_psnr,avg_ssim)
            logging.info(msg)

            logging.info('=> {} validate end, duration: {:.2f}s'.format(head, time.time() - val_start))
            writer.add_scalar('valid_psnr', avg_psnr, epoch)
            writer.add_scalar('valid_ssim', avg_ssim, epoch)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_model(network, final_output_dir, 'best_model.pth')
                writer.add_scalar('best_psnr', best_psnr, epoch)

    writer.close()
    save_model(network, final_output_dir, 'final_model.pth')
    logging.info('=> finish training')
    logging.info("=> Highest PSNR:{:.2f}".format(best_psnr))
