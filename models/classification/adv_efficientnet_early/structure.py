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
            n_filt += endpoints[i].shape[1]

        self.adv_fc1 = nn.Linear(n_filt * 4, 300)
        self.adv_fc2 = nn.Linear(300, 300)
        # self.adv_fc3 = nn.Linear(300, 300)
        self.adv_fc4 = nn.Linear(300, 1)

        # gradient reversal layer
        self.rever1_1 = RevGrad()
        self.rever1_2 = RevGrad()
        self.rever1_3 = RevGrad()
        self.rever1_4 = RevGrad()
        self.rever1_5 = RevGrad()
        self.rever1_6 = RevGrad()
        self.rever1_7 = RevGrad()
        self.rever1_8 = RevGrad()
        self.rever1_9 = RevGrad()
        self.rever1_10 = RevGrad()
        self.rever1_11 = RevGrad()
        self.rever1_12 = RevGrad()

        self.rever2_1 = RevGrad()
        self.rever2_2 = RevGrad()
        self.rever2_3 = RevGrad()
        self.rever2_4 = RevGrad()
        self.rever2_5 = RevGrad()
        self.rever2_6 = RevGrad()
        self.rever2_7 = RevGrad()
        self.rever2_8 = RevGrad()
        self.rever2_9 = RevGrad()
        self.rever2_10 = RevGrad()
        self.rever2_11 = RevGrad()
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
            x = torch.sigmoid(self._fc(x))

        return x, endpoints

    def adversarial_network(self, endpoints, x_s):

        # Convolution layers
        endpoints_s = self.extract_endpoints(x_s)

        x1_s = self.rever1_1(endpoints_s['reduction_1']).mean(dim=2).mean(dim=2)
        x2_s = self.rever1_2(endpoints_s['reduction_2']).mean(dim=2).mean(dim=2)
        x3_s = self.rever1_3(endpoints_s['reduction_3']).mean(dim=2).mean(dim=2)
        x4_s = self.rever1_4(endpoints_s['reduction_4']).mean(dim=2).mean(dim=2)
        x5_s = self.rever1_5(endpoints_s['reduction_5']).mean(dim=2).mean(dim=2)
        x6_s = self.rever1_6(endpoints_s['reduction_6']).mean(dim=2).mean(dim=2)
        x7_s = self.rever1_7(endpoints_s['reduction_1']).std(dim=2).std(dim=2)
        x8_s = self.rever1_8(endpoints_s['reduction_2']).std(dim=2).std(dim=2)
        x9_s = self.rever1_9(endpoints_s['reduction_3']).std(dim=2).std(dim=2)
        x10_s = self.rever1_10(endpoints_s['reduction_4']).std(dim=2).std(dim=2)
        x11_s = self.rever1_11(endpoints_s['reduction_5']).std(dim=2).std(dim=2)
        x12_s = self.rever1_12(endpoints_s['reduction_6']).std(dim=2).std(dim=2)

        x1_p = self.rever2_1(endpoints['reduction_1']).mean(dim=2).mean(dim=2)
        x2_p = self.rever2_2(endpoints['reduction_2']).mean(dim=2).mean(dim=2)
        x3_p = self.rever2_3(endpoints['reduction_3']).mean(dim=2).mean(dim=2)
        x4_p = self.rever2_4(endpoints['reduction_4']).mean(dim=2).mean(dim=2)
        x5_p = self.rever2_5(endpoints['reduction_5']).mean(dim=2).mean(dim=2)
        x6_p = self.rever2_6(endpoints['reduction_6']).mean(dim=2).mean(dim=2)
        x7_p = self.rever2_7(endpoints['reduction_1']).std(dim=2).std(dim=2)
        x8_p = self.rever2_8(endpoints['reduction_2']).std(dim=2).std(dim=2)
        x9_p = self.rever2_9(endpoints['reduction_3']).std(dim=2).std(dim=2)
        x10_p = self.rever2_10(endpoints['reduction_4']).std(dim=2).std(dim=2)
        x11_p = self.rever2_11(endpoints['reduction_5']).std(dim=2).std(dim=2)
        x12_p = self.rever2_12(endpoints['reduction_6']).std(dim=2).std(dim=2)

        x = torch.cat(
            [
                x1_s,
                x2_s,
                x3_s,
                x4_s,
                x5_s,
                x6_s,
                x7_s,
                x8_s,
                x9_s,
                x10_s,
                x11_s,
                x12_s,
                x1_p,
                x2_p,
                x3_p,
                x4_p,
                x5_p,
                x6_p,
                x7_p,
                x8_p,
                x9_p,
                x10_p,
                x11_p,
                x12_p,
            ],
            dim=1,
        )

        x = torch.relu(self.adv_fc1(x))
        # x = torch.relu(self.adv_fc2(x))
        # x = torch.relu(self.adv_fc3(x))
        x = torch.sigmoid(self.adv_fc4(x))

        return x
