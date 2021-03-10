import torch
import torch.nn as nn
from models.acsconv.operators import ACSConv


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ACSConv(in_channels, mid_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)),
            nn.ReLU(inplace=True),
            ACSConv(mid_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvBN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ACSConv(in_channels, mid_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)),
            #nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            ACSConv(mid_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size / 2)),
            #nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv =ACSConv(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1)),DoubleConvBN(in_channels, out_channels, kernel_size, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(2,2,1))
        self.upconv = ACSConv(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size)
        self.conv = DoubleConv(out_channels*2, out_channels, kernel_size, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.upconv(x1)

        # input is CHWD
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2,diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x1,x2],dim=1)
        return self.conv(x)





class UNet(nn.Module):
    def __init__(self, hparams):
        super(UNet, self).__init__()

        self.hparams = hparams

        self.encoder = self.__create_encoder()
        self.decoder = self.__create_decoder()

        self.inc = DoubleConvBN(self.hparams['in_channels'] ,
                    self.hparams['n_filters_input'],
                    self.hparams['kernel_size'],
                    self.hparams['dropout_rate'])
        self.outc = OutConv(self.hparams['n_filters_input'] ,
                    self.hparams['n_classes'])


    def forward(self, x):

        x = self.inc(x)

        encoder_output = []
        encoder_output.append(x)

        for index,layer in enumerate(self.encoder):
            x = layer(x)
            if index < len(self.encoder)-1:
                encoder_output.append(x)

        for index,layer in enumerate(self.decoder):
            x = layer(x,encoder_output[-1*(index+1)])

        x = self.outc(x)
        return torch.nn.functional.softmax(x, dim=1)


    def __create_encoder(self):

        n_layers = self.hparams['n_layers']

        layers = []


        for i in range(n_layers):
            layers.append(
                Down(
                    self.hparams['n_filters_input'] * (2 ** i),
                    self.hparams['n_filters_input'] * (2 ** (i+1)),
                    self.hparams['kernel_size'],
                    self.hparams['dropout_rate'],
                )
            )



        return nn.ModuleList(layers)

    def __create_decoder(self):

        n_layers = self.hparams['n_layers']

        layers = []




        for i in range(n_layers, 0, -1):
            layers.append(
                Up(
                    self.hparams['n_filters_input'] * (2 ** i),
                    self.hparams['n_filters_input'] * (2 ** (i-1)),
                    self.hparams['kernel_size'],
                    self.hparams['dropout_rate'],
                )
            )


        return nn.ModuleList(layers)
