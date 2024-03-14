from pathlib import Path

import os
import logging
import shutil
import time
import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis
import numpy as np

import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from  torchvision.transforms import ToPILImage

from math import exp
import math
import numpy as np
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
def psnr(pred, gt):
    pred=pred.clamp(0,1).cpu().numpy()
    gt=gt.clamp(0,1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10( 1.0 / rmse)
class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr
if __name__ == "__main__":
    pass
def setup_logger(final_output_dir, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.txt'.format(phase, time_str)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s:[P:%(process)d]:' + ' %(message)s'
    logging.basicConfig(
        filename=str(final_log_file), format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)


def create_logger(final_output_dir, phase='train'):
    final_output_dir = Path(final_output_dir)
    print('=> creating {} ...'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    print('=> setup logger ...')
    setup_logger(final_output_dir, phase)


def set_seed_torch(seed=2022):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def summary_model(model, model_name, output_dir, image_size=(256, 256)):
    this_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # copy model file
    shutil.copy2(
        os.path.join(this_dir, 'models', model_name),
        output_dir
    )
    try:
        logging.info('== get_model_complexity_info by thop and fvcore ==')
        input = torch.randn(1, 3, image_size[0], image_size[1])
        flops = FlopCountAnalysis(model, input)
        _, params = profile(model, inputs=(input,))
        flops = flops.total() / 1e9
        params = params / 1e6
        logging.info(f'=> FLOPs: {flops:<8}G, params: {params:<8}M')
        logging.info('== get_model_complexity_info by thop and fvcore ==')
    except Exception:
        logging.error('=> error when run get_model_complexity_info')


def resume_checkpoint(model,
                      optimizer,
                      config,
                      output_dir,
                      in_epoch):
    best_perf = 0.0
    begin_epoch_or_step = 0

    checkpoint = os.path.join(output_dir, 'checkpoint.pth')
    if config.resume_checkpoint and os.path.exists(checkpoint):
        logging.info(
            "=> loading checkpoint '{}'".format(checkpoint)
        )
        checkpoint_dict = torch.load(checkpoint, map_location='cpu')
        best_perf = checkpoint_dict['perf']
        begin_epoch_or_step = checkpoint_dict['epoch' if in_epoch else 'step']
        state_dict = checkpoint_dict['state_dict']
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        logging.info(
            "=> loaded checkpoint '{}' ({}: {})".format(checkpoint, 'epoch' if in_epoch else 'step',
                                                        begin_epoch_or_step)
        )

    return best_perf, begin_epoch_or_step


def save_checkpoint(model,
                    *,
                    model_name,
                    optimizer,
                    output_dir,
                    in_epoch,
                    epoch_or_step,
                    best_perf):
    states = model.state_dict()

    logging.info('=> saving checkpoint to {}'.format(output_dir))
    save_dict = {
        'epoch' if in_epoch else 'step': epoch_or_step + 1,
        'model': model_name,
        'state_dict': states,
        'perf': best_perf,
        'optimizer': optimizer.state_dict(),
    }

    try:
        torch.save(save_dict, os.path.join(output_dir, 'checkpoint.pth'))
    except Exception:
        logging.error('=> error when saving checkpoint!')


def save_model(model, out_dir, fname):
    try:
        fname_full = os.path.join(out_dir, fname)
        logging.info(f'=> save model to {fname_full}')
        torch.save(
            model.state_dict(),
            fname_full
        )
    except Exception:
        logging.error('=> error when saving checkpoint!')
