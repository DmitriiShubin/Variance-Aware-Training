import torch
import torch.nn as nn
from time import time
import numpy as np
from models.pytorch_revgrad import RevGrad


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


class UNet(nn.Module):
    def __init__(self, hparams, bilinear=False):
        super(UNet, self).__init__()

        self.hparams = hparams
        self.n_channels = self.hparams['in_channels']
        self.n_classes = self.hparams['n_classes']
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

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
            self.hparams['n_filters_input'] * 16 // factor,
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
        )
        self.up1 = Up(
            self.hparams['n_filters_input'] * 16,
            self.hparams['n_filters_input'] * 8 // factor,
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
            bilinear,
        )
        self.up2 = Up(
            self.hparams['n_filters_input'] * 8,
            self.hparams['n_filters_input'] * 4 // factor,
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
            bilinear,
        )
        self.up3 = Up(
            self.hparams['n_filters_input'] * 4,
            self.hparams['n_filters_input'] * 2 // factor,
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
            bilinear,
        )
        self.up4 = Up(
            self.hparams['n_filters_input'] * 2,
            self.hparams['n_filters_input'],
            self.hparams['kernel_size'],
            self.hparams['dropout_rate'],
            bilinear,
        )
        self.outc = OutConv(self.hparams['n_filters_input'], self.n_classes)

        # gradient reversal layer
        self.rever1_5 = RevGrad()
        self.rever2_5 = RevGrad()

        n_filt = self.hparams['n_filters_input'] * (2 ** (4))

        self.adv_fc1 = nn.Linear(n_filt, 300)
        self.adv_fc2 = nn.Linear(300, 300)
        self.adv_fc3 = nn.Linear(300, 300)
        self.adv_fc4 = nn.Linear(300, 1)

    def forward(self, x1, x2=None, train=False):

        if train:
            # main head (predictive)
            out, decoder_x = self.predictive_network(x1)
            # additional head (adversarial)
            out_s = self.adversarial_network(decoder_x, x2)
            return out, out_s
        else:
            # main head (predictive)
            out, _ = self.predictive_network(x1)
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

        x5_s = self.rever1_5(x5).mean(dim=2).mean(dim=2)

        x5_p = self.rever2_5(x[4]).mean(dim=2).mean(dim=2)

        x = torch.cat([x5_s, x5_p], dim=1)

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
