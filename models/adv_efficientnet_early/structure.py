import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet as effnet
from models.pytorch_revgrad import RevGrad

class EfficientNet(effnet):
    def __init__(self, hparams):
        super(EfficientNet, self).__init__()

        self.from_pretrained(hparams['pre_trained_model'], num_classes=hparams['n_classes'])

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

    def forward(self, x1, x2=None, train=False):

        if train:
            # main head (predictive)
            out, encoder_x = self.predictive_network(x1)
            # additional head (adversarial)
            out_s = self.adversarial_network(encoder_x, x2)
            return out, out_s
        else:
            # main head (predictive)
            out, _ = self.predictive_network(x1)
            return out

    def adversarial_network(self, featurs, x_s):
        featurs_s = self.extract_endpoints(x_s)

        x1_s = self.rever1_1(featurs_s['reduction_1']).mean(dim=2).mean(dim=2)
        x2_s = self.rever1_2(featurs_s['reduction_2']).mean(dim=2).mean(dim=2)
        x3_s = self.rever1_3(featurs_s['reduction_3']).mean(dim=2).mean(dim=2)
        x4_s = self.rever1_4(featurs_s['reduction_4']).mean(dim=2).mean(dim=2)
        x5_s = self.rever1_5(featurs_s['reduction_5']).mean(dim=2).mean(dim=2)
        x6_s = self.rever1_6(featurs_s['reduction_6']).mean(dim=2).mean(dim=2)
        x7_s = self.rever1_7(featurs_s['reduction_1']).std(dim=2).std(dim=2)
        x8_s = self.rever1_8(featurs_s['reduction_2']).std(dim=2).std(dim=2)
        x9_s = self.rever1_9(featurs_s['reduction_3']).std(dim=2).std(dim=2)
        x10_s = self.rever1_10(featurs_s['reduction_4']).std(dim=2).std(dim=2)
        x11_s = self.rever1_11(featurs_s['reduction_5']).std(dim=2).std(dim=2)
        x12_s = self.rever1_12(featurs_s['reduction_6']).std(dim=2).std(dim=2)

        x1_p = self.rever2_1(featurs['reduction_1']).mean(dim=2).mean(dim=2)
        x2_p = self.rever2_2(featurs['reduction_2']).mean(dim=2).mean(dim=2)
        x3_p = self.rever2_3(featurs['reduction_3']).mean(dim=2).mean(dim=2)
        x4_p = self.rever2_4(featurs['reduction_4']).mean(dim=2).mean(dim=2)
        x5_p = self.rever2_5(featurs['reduction_5']).mean(dim=2).mean(dim=2)
        x6_p = self.rever1_6(featurs['reduction_6']).mean(dim=2).mean(dim=2)
        x7_p = self.rever2_7(featurs['reduction_1']).std(dim=2).std(dim=2)
        x8_p = self.rever2_8(featurs['reduction_2']).std(dim=2).std(dim=2)
        x9_p = self.rever2_9(featurs['reduction_3']).std(dim=2).std(dim=2)
        x10_p = self.rever2_10(featurs['reduction_4']).std(dim=2).std(dim=2)
        x11_p = self.rever2_11(featurs['reduction_5']).std(dim=2).std(dim=2)
        x12_p = self.rever1_12(featurs['reduction_6']).std(dim=2).std(dim=2)

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

    def predictive_network(self, x):
        featurs = self.extract_endpoints(x)
        x = self.extract_features(x)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            logits = self._fc(x)

            return logits,featurs
        else:
            return x,featurs