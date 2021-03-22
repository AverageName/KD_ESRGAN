import glob
import random
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

def crop_image(image, i, j, h, w):
    return image[:, i: i+h, j:j+w]

def transform_to_torch(image):
  image = image * 1.0 / 255.0
  if len(image.shape) == 2:
      image = np.stack([image, image, image], axis=-1)
      image_orig = np.stack([image_orig, image_orig, image_orig], axis=-1)

  img = torch.from_numpy(np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))).float()

  return img

def transform_torch(img):
    i, j, h, w = transforms.RandomCrop.get_params(img, (128, 128))
    height, width = img.shape[1], img.shape[2]
    i = min(max(i, 128), height - 128)
    j = min(max(j, 128), width - 128)
    crop = crop_image(img, i, j, h, w)

    return crop, (i, j, h, w)

class ImageDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_COLOR)
        img_lr = transform(img)

        return img_lr

    def __len__(self):
        return len(self.files)


class DIV2K(Dataset):

    def __init__(self, root, train=True, distill=True, num_examples=None):
        self.distill = distill

        if train:
            self.lr = sorted(glob.glob(os.path.join(root, '*train_LR_bicubic/X4/*.png')))
        else:
            self.lr = sorted(glob.glob(os.path.join(root, '*valid_LR_bicubic/X4/*.png')))
        
        if not distill:
            if train:
                self.hr = sorted(glob.glob(os.path.join(root, '*train_HR/*.png')))
            else:
                self.hr = sorted(glob.glob(os.path.join(root, '*valid_HR/*.png')))
        
        if num_examples is not None:
            self.lr = self.lr[:num_examples]
            if not distill:
              self.hr = self.hr[:num_examples]
        
    def __len__(self):
        return len(self.lr)
    
    def __getitem__(self, idx):
        img_lr = cv2.imread(self.lr[idx % len(self.lr)], cv2.IMREAD_COLOR)
        img_lr = transform_to_torch(img_lr)

        if not self.distill:
            img_hr = cv2.imread(self.hr[idx % len(self.hr)], cv2.IMREAD_COLOR)
            img_hr = transform_to_torch(img_hr)

            img_lr, coords = transform_torch(img_lr)
            img_hr = crop_image(img_hr, coords[0]*4, coords[1]*4, coords[2]*4, coords[3]*4)
            

            return img_lr, img_hr
        else:
            crop, _ = transform_torch(img_lr)
            return crop










