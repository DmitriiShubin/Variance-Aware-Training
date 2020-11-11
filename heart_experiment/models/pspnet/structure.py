import torch
import torch.nn as nn
from segmentation_models_pytorch import PSPNet as smp_PSPNet


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PSPNet(smp_PSPNet):
    def __init__(self, hparams, n_channels, n_classes):
        depth = 4
        super(PSPNet, self).__init__(
            encoder_name='resnet18',
            encoder_depth=depth,
            encoder_weights=None,
            in_channels=n_channels,
            psp_use_batchnorm=True, #optional?
            psp_dropout=0.0,
            classes=n_classes,
            upsampling=16
        )


        self.hparams = hparams['model']



    def forward(self, x):
        return self.predictive_network(x)

    def predictive_network(self, x):
        features = self.encoder(x)
        x = self.decoder(*features)
        x = self.segmentation_head(x)

        logits = torch.nn.functional.softmax(x, dim=1)
        return logits

