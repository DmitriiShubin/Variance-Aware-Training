import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class FPN(nn.Module):
    def __init__(self, n_filter_1, n_filter_2, n_filter_3):
        """Origin: https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py"""
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Smooth layers
        self.smooth1 = nn.Conv2d(n_filter_1, n_filter_1, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(n_filter_1, n_filter_1, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(n_filter_1, n_filter_1, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(n_filter_3, n_filter_1, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(n_filter_2, n_filter_1, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(n_filter_1, n_filter_1, kernel_size=1, stride=1, padding=0)

        self.activation = Mish()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, c2, c3, c4):

        # Top-down

        p4 = self.activation(self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.activation(self.latlayer2(c3)))
        p2 = self._upsample_add(p3, self.activation(self.latlayer3(c2)))
        # Smooth
        p4 = self.activation(self.smooth1(p4))
        p3 = self.activation(self.smooth2(p3))
        p2 = self.activation(self.smooth3(p2))
        return p2, p3, p4
