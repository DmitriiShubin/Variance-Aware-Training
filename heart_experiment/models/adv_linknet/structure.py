import torch
import torch.nn as nn
from segmentation_models_pytorch import Linknet as smp_linknet


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LinkNet(smp_linknet):
    def __init__(self, hparams, n_channels, n_classes):
        depth = 5
        super(LinkNet, self).__init__(
            encoder_name='resnet34',
            encoder_depth=depth,
            encoder_weights=None,
            in_channels=n_channels,
            decoder_use_batchnorm=True, #optional?
            classes=1,
        )

        self.hparams = hparams['model']

        self.outc = OutConv(32, n_classes)

        # adversarial deep net layers
        self.adv_conv1 = nn.Conv2d(int(32 * (2**depth)), 1, kernel_size=1, padding=0)
        self.adv_fc1 = nn.Linear(int(320/(2**depth)), 1)
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

        logits = torch.nn.functional.softmax(self.outc(x), dim=1)
        return logits, features[-1]

