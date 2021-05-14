import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet as effnet


class EfficientNet(effnet):
    def __init__(self, blocks_args=None, global_params=None):

        super(EfficientNet, self).__init__(blocks_args=blocks_args, global_params=global_params)

    def forward(self, inputs1, inputs2=None, pretrain=False):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x1 = self.extract_features(inputs1)
        if pretrain:
            x2 = self.extract_features(inputs2)

            x = torch.cat([x1, x2], dim=1)

            x = torch.mean(x, dim=2)
            x = torch.mean(x, dim=2)

            x = torch.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))

            return x
        else:
            return x1

    def build_projection_network(self, device):

        dummy = torch.rand(1, 3, 96, 96).to(device)
        n_filt = self.extract_features(dummy).shape[1]

        self.fc1 = nn.Linear(n_filt * 2, n_filt * 2).to(device)
        self.fc2 = nn.Linear(n_filt * 2, 1).to(device)

        return True
