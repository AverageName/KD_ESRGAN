import os
import functools
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import pytorch_lightning as pl
import math
import argparse
import numpy as np

from torchvision.models import vgg19
from utils import criterions, calc_psnr, calc_ssim, torch2image
from PIL import Image


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        embeds = []
        fea = self.conv_first(x)
        embeds.append(fea)
        out = fea
        for block in self.RRDB_trunk:
            out = block(out)
            embeds.append(out)
        trunk = self.trunk_conv(out)
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out, embeds


class RRDBKDNet(pl.LightningModule):

  def __init__(self, hparams):
    super().__init__()

    if isinstance(hparams, dict):
        hparams = argparse.Namespace(**hparams)

    self.hparams = hparams
    self.teacher_model = RRDBNet(hparams.in_nc, hparams.out_nc,
                         hparams.nf, hparams.nb, hparams.gc)
    self.teacher_model.load_state_dict(torch.load(hparams.orig_checkpoint_path))
    self.teacher_model.eval()
    
    self.student_model = RRDBNet(hparams.in_nc, hparams.out_nc,
                                 hparams.nf // hparams.nf_shrink, hparams.nb // hparams.nb_shrink,
                                 hparams.gc // hparams.gc_shrink)
    print(type(self.hparams.train_criterions))
    self.train_criterions = [(criterions(criterion), weight, criterion) for criterion, weight in self.hparams.train_criterions.items()]
    self.val_criterions = [(criterions(criterion), weight, criterion) for criterion, weight in self.hparams.val_criterions.items()]

    if hparams.feature_matching is not None:
        self.feature_matching, self.feature_matching_weight = hparams.feature_matching

    if hparams.gt_criterions is not None:
        self.gt_criterions = [(criterions(criterion), weight, criterion) for criterion, weight in self.hparams.gt_criterions.items()]
  
  def forward(self, x):
    return self.student_model(x)
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.student_model.parameters(), self.hparams.lr)
    return optimizer
  
  def training_step(self, batch, batch_idx):
    with torch.no_grad():
      gt, features_gt = self.teacher_model(batch[0])
    predict, features_pred = self.student_model(batch[0])

    # loss_teaching = self.criterion(predict, gt)
    loss = 0.0
    for criterion, weight, name in self.train_criterions:
        if name == "ms_ssim":
            criterion_loss = criterion(gt.clamp_(0, 1), predict.clamp_(0, 1))
            criterion_loss = 1 - criterion_loss
        else:
            criterion_loss = criterion(predict, gt)
            if name == "lpips":
                criterion_loss = torch.sum(criterion_loss)

        loss += weight * criterion_loss
        self.log(name, criterion_loss)

    if self.feature_matching:
        feature_matching_loss = 0.0
        counter = 0
        for i in range(1, len(features_pred) - 1):
            if counter % self.hparams.nb_shrink:
                feature_matching_loss += F.mse_loss(features_pred[i], features_gt[self.hparams.nb_shrink * i])
            counter += 1
        feature_matching_loss += F.mse_loss(features_pred[0], features_gt[0])
        feature_matching_loss += F.mse_loss(features_pred[-1], features_gt[-1])

        self.log("feature_mathing_loss", feature_matching_loss)
        loss += self.feature_matching_weight * feature_matching_loss


    if self.gt_criterions:
        for criterion, weight, name in self.gt_criterions:
            if name == "ms_ssim":
                criterion_loss = criterion(batch[1].clamp_(0, 1), predict.clamp_(0, 1))
                criterion_loss = 1 - criterion_loss
            else:
                criterion_loss = criterion(predict, batch[1])
                if name == "lpips":
                    criterion_loss = torch.sum(criterion_loss)

            loss += weight * criterion_loss
            self.log(name + "_gt", criterion_loss)

    self.log("train_batch_loss", loss)
    return {"loss": loss}

  def validation_step(self, batch, batch_idx):

    predict, _ = self.student_model(batch[0])
    gt = batch[1]

    predict_numpy = torch2image(predict)[:, :, ::-1]
    gt_numpy = torch2image(gt)[:, :, ::-1]
    psnr = calc_psnr(predict_numpy, gt_numpy)
    # loss = 0.0
    # for criterion, weight, name in self.val_criterions:
    #     criterion_loss = criterion(predict, gt)
    #     loss += weight * criterion_loss
    #     self.log(name, criterion_loss)

    return {"origs": batch[0], 'psnr': psnr, "gt": batch[1], "predict": predict}
  
  def validation_epoch_end(self, outputs):
    mean_psnr = np.mean([item['psnr'] for item in outputs])
    self.log('val_psnr', mean_psnr)
    logger = self.logger.experiment[0]
    teacher_images = torch.cat([item["gt"] for item in outputs[-4:]], dim=0)
    student_images = torch.cat([item["predict"] for item in outputs[-4:]], dim=0)
    viz_batch = torch.cat([teacher_images, student_images], dim=0)
    grid = torchvision.utils.make_grid(viz_batch)
    grid = torch2image(grid)[:, :, ::-1]
    print(grid.shape)
    logger.log_image(image_data=grid, name=str(self.current_epoch) + "_hr")

    lr_images = torch.cat([item["origs"] for item in outputs[-4:]], dim=0)
    grid = torchvision.utils.make_grid(lr_images)
    grid = torch2image(grid)[:, :, ::-1]
    print(grid.shape)
    logger.log_image(image_data=grid, name=str(self.current_epoch) + "_lr")

  def test_step(self, batch, batch_idx):
    hr_pred, _ = self.student_model(batch[0])
    hr_pred_teacher, _ = self.teacher_model(batch[0])
    hr_gt = batch[1]

    lr_numpy = torch2image(batch[0])[:, :, ::-1]
    hr_teacher_numpy = torch2image(hr_pred_teacher)[:, :, ::-1]
    hr_pred_numpy = torch2image(hr_pred)[:, :, ::-1]
    hr_gt_numpy = torch2image(hr_gt)[:, :, ::-1]

    psnr = calc_psnr(hr_gt_numpy, hr_pred_numpy)
    ssim = calc_ssim(hr_gt_numpy, hr_pred_numpy)

    if not os.path.exists("./results/"):
        os.makedirs("./results/")

    try:
        logger = self.logger.experiment[0]
        name = logger.get_key()
    except:
        name = "tmp"

    if not os.path.exists("./results/{}".format(name)):
        os.makedirs("./results/{}".format(name))

    lr_image = Image.fromarray(lr_numpy)
    hr_teacher = Image.fromarray(hr_teacher_numpy)
    img_pred = Image.fromarray(hr_pred_numpy)
    gt_pred = Image.fromarray(hr_gt_numpy)

    img_pred.save("./results/{}/{}_pred.png".format(name, str(batch_idx)), format="PNG")
    hr_teacher.save("./results/{}/{}_teacher.png".format(name, str(batch_idx)), format="PNG")
    gt_pred.save("./results/{}/{}_gt.png".format(name, str(batch_idx)), format="PNG")
    lr_image.save("./results/{}/{}_lr.png".format(name, str(batch_idx)), format="PNG")

    self.log_dict({"psnr": psnr, "ssim": ssim})


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

class ESRGAN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.hparams = hparams

        self.g = RRDBNet(hparams.in_nc, hparams.out_nc,
                         hparams.nf, hparams.nb, hparams.gc)
        hparams.orig_checkpoint_path = '/content/drive/My Drive/RRDB_PSNR_x4.pth'
        self.g.load_state_dict(torch.load(hparams.orig_checkpoint_path))
        self.g.eval()

        self.d = Discriminator((3, 128, 128))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.d.parameters(), self.hparams.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        lr = batch[0]
        real = batch[1]

        with torch.no_grad():
            fake = self.g(lr)
        d_fake = self.d(fake)
        d_real = self.d(real)

        loss = - torch.mean(torch.log(torch.sigmoid((d_real.squeeze()[0] - torch.mean(d_fake.squeeze()))))) - torch.mean(torch.log(1 - torch.sigmoid((d_fake.squeeze()[0] - torch.mean(d_real.squeeze())))))
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        pass











        




