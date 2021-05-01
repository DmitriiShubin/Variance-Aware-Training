import torch
import torch.nn as nn
from time import time
import numpy as np
from efficientnet_pytorch import EfficientNet as effnet


class EfficientNet(effnet):
    def __init__(self, blocks_args=None, global_params=None):

        super(EfficientNet, self).__init__(blocks_args=blocks_args, global_params=global_params)

    def build_projection_network(self, emb_dim, device):

        dummy = torch.rand(1, 3, 96, 96).to(device)
        n_filt = self.extract_features(dummy).shape[1]

        self.fc1 = nn.Linear(n_filt, n_filt).to(device)
        self.fc2 = nn.Linear(n_filt, n_filt).to(device)
        self.fc3 = nn.Linear(n_filt, emb_dim).to(device)

        return True

    def forward(self, x):
        x = self.extract_features(x)

        x = torch.mean(x, dim=2)
        x = torch.mean(x, dim=2)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def encoder(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        return x1, x2, x3, x4, x5, x6
