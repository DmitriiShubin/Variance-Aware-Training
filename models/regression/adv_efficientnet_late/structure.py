import torch
import torch.nn as nn
from models.pytorch_revgrad import RevGrad

from efficientnet_pytorch import EfficientNet as effnet


class EfficientNet(effnet):
    def __init__(self, blocks_args=None, global_params=None):

        super(EfficientNet, self).__init__(blocks_args=blocks_args, global_params=global_params)

    def forward(self, x1, x2=None, train=False):

        if train:
            # main head (predictive)
            out, endpoints = self.predictive_network(x1)
            # additional head (adversarial)
            out_s = self.adversarial_network(endpoints, x2)
            return out, out_s
        else:
            # main head (predictive)
            out, _ = self.predictive_network(x1)
            return out

    def build_adv_model(self):

        dummy = torch.rand(1, 3, 96, 96)
        endpoints = self.extract_endpoints(dummy)

        n_filt = 0
        for i in endpoints.keys():
            if i == 'reduction_6':
                n_filt += endpoints[i].shape[1]

        self.adv_fc1 = nn.Linear(n_filt * 4, 300)
        self.adv_fc2 = nn.Linear(300, 300)
        # self.adv_fc3 = nn.Linear(300, 300)
        self.adv_fc4 = nn.Linear(300, 1)

        # gradient reversal layer
        self.rever1_6 = RevGrad()
        self.rever1_12 = RevGrad()

        self.rever2_6 = RevGrad()
        self.rever2_12 = RevGrad()

        return True

    def predictive_network(self, inputs):

        # Convolution layers
        endpoints = self.extract_endpoints(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(endpoints['reduction_6'])
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)

        return x, endpoints

    def adversarial_network(self, endpoints, x_s):

        # Convolution layers
        endpoints_s = self.extract_endpoints(x_s)

        x6_s = self.rever2_6(endpoints_s['reduction_6']).mean(dim=2).mean(dim=2)
        x12_s = self.rever2_12(endpoints_s['reduction_6']).std(dim=2).std(dim=2)

        x6_p = self.rever2_6(endpoints['reduction_6']).mean(dim=2).mean(dim=2)
        x12_p = self.rever2_12(endpoints['reduction_6']).std(dim=2).std(dim=2)

        x = torch.cat([x6_s, x12_s, x6_p, x12_p,], dim=1,)

        x = torch.relu(self.adv_fc1(x))
        x = torch.sigmoid(self.adv_fc4(x))

        return x
