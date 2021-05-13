import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet as effnet


class EfficientNet(effnet):
    def __init__(self, blocks_args=None, global_params=None):

        super(EfficientNet, self).__init__(blocks_args=blocks_args, global_params=global_params)

    def forward(self, inputs,meta):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(x)

        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = torch.cat([x, meta], dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    def freeze_layers(self):

        for param in self.parameters():
            print(param)
            param.requires_grad = False

        return True

    def build_out_fc(self):

        dummy = torch.rand(1, 3, 96, 96)
        endpoints = self.extract_features(dummy)

        self.fc_ = nn.Linear(endpoints.shape[1]+3,1)

        return True