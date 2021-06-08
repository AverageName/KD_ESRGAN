import numpy as np
import cv2 
import torch
import torch.nn as nn
import lpips

from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_ssim
from pytorch_msssim import MS_SSIM

def criterions(name):
    if name == "mse":
        return nn.MSELoss()
    elif name == "l1":
        return nn.L1Loss()
    elif name == "lpips":
        return lpips.LPIPS(net="vgg").cuda()
    elif name == "ms_ssim":
        return MS_SSIM(data_range=1.0)


def torch2image(tensor):
    result = tensor.cpu().squeeze().detach().float().clamp_(0, 1).numpy()
    output = np.transpose(result[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    return output


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float64)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def calc_psnr(img, rec):
  return cv2.PSNR(rgb2ycbcr(img), rgb2ycbcr(rec))


def calc_ssim(img, rec):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  rec = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)

  return compare_ssim(img, rec)
