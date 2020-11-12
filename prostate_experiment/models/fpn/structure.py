import torch
import torch.nn as nn
from time import time
import numpy as np
from segmentation_models_pytorch import FPN as smp_FPN


class FPN(smp_FPN):
    def __init__(self, hparams, n_channels, n_classes, bilinear=True):
        super(FPN, self).__init__(
            # encoder_depth=5,
            encoder_weights=None,
            # decoder_dropout=0.0,
            # decoder_pyramid_channels=256,
            # decoder_segmentation_channels=128,
            # in_channels=n_channels,
            # classes=n_classes
        )


        #self.outc = OutConv(self.hparams['n_filters_input'], n_classes)



    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4, x5)
        logits = self.outc(x)
        logits = torch.nn.functional.softmax(logits,dim=1)
        return logits

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
