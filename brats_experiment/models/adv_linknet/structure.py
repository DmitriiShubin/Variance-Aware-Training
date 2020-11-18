import torch
import torch.nn as nn
from time import time
import numpy as np
from utils.pytorch_revgrad import RevGrad

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, kernel_size, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        #if bilinear:
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout, in_channels // 2)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        #x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x1 = self.conv(x1)
        return torch.add(x2, x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LinkNet(nn.Module):
    def __init__(self, hparams, n_channels, n_classes, bilinear=True):
        super(LinkNet, self).__init__()

        self.hparams = hparams['model']
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(
            n_channels, self.hparams['n_filters_input'], self.hparams['kernel_size'], self.hparams['dropout']
        )
        self.down1 = Down(
            self.hparams['n_filters_input'],
            self.hparams['n_filters_input'] * 2,
            self.hparams['kernel_size'],
            self.hparams['dropout'],
        )
        self.down2 = Down(
            self.hparams['n_filters_input'] * 2,
            self.hparams['n_filters_input'] * 4,
            self.hparams['kernel_size'],
            self.hparams['dropout'],
        )
        self.down3 = Down(
            self.hparams['n_filters_input'] * 4,
            self.hparams['n_filters_input'] * 8,
            self.hparams['kernel_size'],
            self.hparams['dropout'],
        )

        self.down4 = Down(
            self.hparams['n_filters_input'] * 8,
            self.hparams['n_filters_input'] * 16,
            self.hparams['kernel_size'],
            self.hparams['dropout'],
        )
        self.up1 = Up(
            self.hparams['n_filters_input'] * 16,
            self.hparams['n_filters_input'] * 8,
            self.hparams['kernel_size'],
            self.hparams['dropout'],
            bilinear,
        )
        self.up2 = Up(
            self.hparams['n_filters_input'] * 8,
            self.hparams['n_filters_input'] * 4,
            self.hparams['kernel_size'],
            self.hparams['dropout'],
            bilinear,
        )
        self.up3 = Up(
            self.hparams['n_filters_input'] * 4,
            self.hparams['n_filters_input'] * 2,
            self.hparams['kernel_size'],
            self.hparams['dropout'],
            bilinear,
        )
        self.up4 = Up(
            self.hparams['n_filters_input'] * 2,
            self.hparams['n_filters_input'],
            self.hparams['kernel_size'],
            self.hparams['dropout'],
            bilinear,
        )
        self.outc = OutConv(self.hparams['n_filters_input'], n_classes)

        #gradient reversal layer
        self.rever1_1 = RevGrad()
        self.rever1_2 = RevGrad()
        self.rever1_3 = RevGrad()
        self.rever1_4 = RevGrad()
        self.rever1_5 = RevGrad()
        self.rever2_1 = RevGrad()
        self.rever2_2 = RevGrad()
        self.rever2_3 = RevGrad()
        self.rever2_4 = RevGrad()
        self.rever2_5 = RevGrad()

        # adversarial deep net layers
        self.adv_conv1 = nn.Conv2d(self.hparams['n_filters_input'] * 32, self.hparams['n_filters_input'] * 32, kernel_size=1, padding=0)
        self.adv_conv2 = nn.Conv2d(self.hparams['n_filters_input'] * 32, 1, kernel_size=1, padding=0)
        self.adv_fc1 = nn.Linear(992, 300)
        self.adv_fc2 = nn.Linear(300, 300)
        self.adv_fc3 = nn.Linear(300, 20)
        self.adv_fc4 = nn.Linear(20, 1)
        # self.adv_fc2 = nn.Linear(self.hparams['n_filters_input'], 1)

    def forward(self, x, adv_head=True):
        x, x_s = x  # unpack training and adversarial images

        # main head (predictive)
        out, decoder_x = self.predictive_network(x)


        if adv_head:
            # additional head (adversarial)

            out_s = self.adversarial_network(decoder_x, x_s)

            return out, out_s
        else:
            return out

    def encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5

    def decoder(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def adversarial_network(self, x, x_s):
        x1, x2, x3, x4, x5 = self.encoder(x_s)
        # x_s = self.decoder(x1, x2, x3, x4, x5)

        x1_s = self.rever1_1(x1).mean(dim=2).mean(dim=2)
        x2_s = self.rever1_2(x2).mean(dim=2).mean(dim=2)
        x3_s = self.rever1_3(x3).mean(dim=2).mean(dim=2)
        x4_s = self.rever1_4(x4).mean(dim=2).mean(dim=2)
        x5_s = self.rever1_5(x5).mean(dim=2).mean(dim=2)

        x1_p = self.rever2_1(x[0]).mean(dim=2).mean(dim=2)
        x2_p = self.rever2_2(x[1]).mean(dim=2).mean(dim=2)
        x3_p = self.rever2_3(x[2]).mean(dim=2).mean(dim=2)
        x4_p = self.rever2_4(x[3]).mean(dim=2).mean(dim=2)
        x5_p = self.rever2_5(x[4]).mean(dim=2).mean(dim=2)

        x = torch.cat([x1_s, x2_s, x3_s, x4_s, x5_s, x1_p, x2_p, x3_p, x4_p, x5_p], dim=1)

        x = torch.relu(self.adv_fc1(x))
        x = torch.relu(self.adv_fc2(x))
        x = torch.relu(self.adv_fc3(x))
        x = torch.sigmoid(self.adv_fc4(x))

        return x

    def predictive_network(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4, x5)
        logits = self.outc(x)
        logits = torch.nn.functional.softmax(logits, dim=1)
        return logits, [x1, x2, x3, x4, x5]