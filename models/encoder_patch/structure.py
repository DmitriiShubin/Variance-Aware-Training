import torch
import torch.nn as nn
from time import time
import numpy as np


class DoubleConvBN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x):

        x = self.bn1(torch.relu(self.conv1(x)))
        x = x = self.drop1(x)
        identity_full = x

        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.drop2(x)
        x += identity_full
        identity_1 = x

        x = self.bn3(torch.relu(self.conv3(x)))
        x = x = self.drop3(x)
        x += identity_full
        x += identity_1

        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)
        )

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x):

        x = self.drop1(torch.relu(self.conv1(x)))
        identity_full = x

        x = self.drop2(torch.relu(self.conv2(x)))
        x += identity_full
        identity_1 = x

        x = self.drop3(torch.relu(self.conv3(x)))
        x += identity_full
        x += identity_1

        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConvBN(in_channels, out_channels, kernel_size, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout)  # , in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class Encoder_patch(nn.Module):
    def __init__(self, hparams, bilinear=False):
        super(Encoder_patch, self).__init__()

        self.hparams = hparams
        self.n_channels = self.hparams['in_channels']
        self.emb_dim = self.hparams['emb_dim']
        self.bilinear = bilinear

        self.factor = 2 if bilinear else 1

        self.inc = DoubleConv(
            self.n_channels,
            self.hparams['n_filters_input'],
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
        )
        self.down1 = Down(
            self.hparams['n_filters_input'],
            self.hparams['n_filters_input'] * 2,
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
        )
        self.down2 = Down(
            self.hparams['n_filters_input'] * 2,
            self.hparams['n_filters_input'] * 4,
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
        )
        self.down3 = Down(
            self.hparams['n_filters_input'] * 4,
            self.hparams['n_filters_input'] * 8,
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
        )

        self.down4 = Down(
            self.hparams['n_filters_input'] * 8,
            self.hparams['n_filters_input'] * 16 // self.factor,
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
        )
        self.down5 = Down(
            self.hparams['n_filters_input'] * 16,
            self.hparams['n_filters_input'] * 32 // self.factor,
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
        )

        self.fc1 = nn.Linear(
            self.hparams['n_filters_input'] * (2 ** 5)*2, self.hparams['n_filters_input'] * (2 ** 5)*2
        )
        self.fc2 = nn.Linear(
            self.hparams['n_filters_input'] * (2 ** 5)*2, self.hparams['n_classes']
        )
        # self.fc3 = nn.Linear(self.hparams['n_filters_input'] * (2 ** 5), 128)#self.emb_dim)

    def forward(self, x1,x2):
        _, _, _, _, _, x1 = self.encoder(x1)
        _, _, _, _, _, x2 = self.encoder(x2)

        x = torch.cat([x1,x2],dim=1)

        x = torch.mean(x, dim=2)
        x = torch.mean(x, dim=2)

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # logits = self.fc3(x)
        return x

    def encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        return x1, x2, x3, x4, x5, x6
