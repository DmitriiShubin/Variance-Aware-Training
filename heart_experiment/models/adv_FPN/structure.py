import torch
import torch.nn as nn
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
            encoder_depth=4,
            encoder_weights=None,
            decoder_pyramid_channels=hparams['model']['n_filters_input'],
            decoder_segmentation_channels=hparams['model']['n_filters_input'],
            decoder_dropout=hparams['model']['dropout'],
            in_channels=n_channels,
        )

        self.conv2d = nn.Conv2d(
            hparams['model']['n_filters_input'],
            hparams['model']['n_filters_input'],
            kernel_size=1,
            padding=0,
        )
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.hparams = hparams['model']

        # self.outc = OutConv(self.hparams['n_filters_input'], n_classes)

        # adversarial deep net layers
        # self.adv_fc1 = nn.Linear(self.hparams['n_filters_input'], 1)

        self.outc = OutConv(self.hparams['n_filters_input'], n_classes)

        # adversarial deep net layers
        self.adv_conv1 = nn.Conv2d(self.hparams['n_filters_input'] * 16, 1, kernel_size=1, padding=0)
        self.adv_fc1 = nn.Linear(20, 1)
        #self.adv_fc2 = nn.Linear(self.hparams['n_filters_input'], 1)

    # def forward(self, x):
    #     x1, x2, x3, x4, x5 = self.encoder(x)
    #     x = self.decoder(x1, x2, x3, x4, x5)
    #     logits = self.outc(x)
    #     logits = torch.softmax(logits, dim=1)
    #     return logits

    def forward(self, x):
        x, x_s = x  # unpack 2 pictures
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # main head (predictive)
        out, decoder_x = self.predictive_network(x)

        # additional head (adversarial)
        out_s = self.adversarial_network(decoder_x, x_s)

        return out, out_s

    def adversarial_network(self, x, x_s):

        features = self.encoder(x_s)
        x_s = features[-1]



        x = torch.cat((x, x_s), dim=1)

        x = torch.relu(self.adv_conv1(x))

        x = torch.mean(x, dim=2)
        x = torch.squeeze(x)
        x = torch.sigmoid(self.adv_fc1(x))
        return x

    def predictive_network(self, x):
        features = self.encoder(x)
        x = self.decoder(*features)
        x = self.conv2d(x)
        x = self.upsampling(x)
        logits = self.outc(x)
        logits = torch.nn.functional.softmax(logits, dim=1)
        return logits, features[-1]

    # def adversarial_network(self, x, x_s):
    #
    #     x1, x2, x3, x4, x5 = self.encoder(x_s)
    #     #x_s = self.decoder(x1, x2, x3, x4, x5)
    #
    #
    #
    #     x = torch.cat((x, x5), dim=1)
    #
    #     x = torch.relu(self.adv_conv1(x))
    #
    #     x = torch.mean(x, dim=2)
    #     x = torch.squeeze(x)
    #     x = torch.sigmoid(self.adv_fc1(x))
    #     return x
    #
    # def predictive_network(self, x):
    #     x1, x2, x3, x4, x5 = self.encoder(x)
    #     x = self.decoder(x1, x2, x3, x4, x5)
    #     logits = self.outc(x)
    #     logits = torch.nn.functional.softmax(logits, dim=1)
    #     return logits, x5

