import torch
import torch.nn as nn
from loss_functions import AngularPenaltySMLoss
from segmentation_models_pytorch import FPN as smp_FPN

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FPN(smp_FPN):
    def __init__(self, hparams, n_channels, n_classes):
        super(FPN, self).__init__(

            encoder_name='resnet34',
            encoder_depth = 4,
            encoder_weights=None,
            decoder_pyramid_channels=hparams['model']['n_filters_input'],
            decoder_segmentation_channels=hparams['model']['n_filters_input'],
            decoder_dropout=hparams['model']['dropout'],
            in_channels=n_channels,

        )

        self.conv2d = nn.Conv2d(hparams['model']['n_filters_input'], hparams['model']['n_filters_input'], kernel_size=1, padding=0)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.hparams = hparams['model']



        self.outc = OutConv(self.hparams['n_filters_input'], n_classes)

        # adversarial deep net layers
        self.adv_fc1 = nn.Linear(self.hparams['n_filters_input'], 1)

    # def forward(self, x):
    #     x1, x2, x3, x4, x5 = self.encoder(x)
    #     x = self.decoder(x1, x2, x3, x4, x5)
    #     logits = self.outc(x)
    #     logits = torch.softmax(logits, dim=1)
    #     return logits

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)
        x = self.decoder(*features)

        x = self.conv2d(x)
        x = self.upsampling(x)

        x = self.outc(x)

        x = torch.softmax(x,dim=1)

        return x