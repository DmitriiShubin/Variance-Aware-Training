import torch
import torch.nn as nn
import numpy as np
from models.adv_FasterRCNN_late.EfficientNet import EfficientNet
import torchvision
from torchvision.models.detection import FasterRCNN as FasterRCNN_torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from models.pytorch_revgrad import RevGrad


class FasterRCNN(nn.Module):
    def __init__(self, hparams, device):
        super(FasterRCNN, self).__init__()

        self.hparams = hparams
        self.backbone = EfficientNet.from_pretrained(self.hparams['pre_trained_model'])

        dummy = torch.rand(1, 3, 96, 96)
        endpoints = self.backbone.extract_endpoints(dummy)
        n_filt = endpoints['reduction_6'].shape[1]
        self.backbone.out_channels = n_filt
        self.backbone.to(device)

        self.anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        ).to(device)
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        ).to(device)

        self.frcnn = FasterRCNN_torchvision(
            self.backbone,
            num_classes=self.hparams['n_classes'],
            rpn_anchor_generator=self.anchor_generator,
            box_roi_pool=self.roi_pooler,
        ).to(device)

    def forward(self, x1, x2=None, target=None, train=False):

        if train:
            # main head (predictive)
            loss, endpoints = self.predictive_network(x1, target)
            # additional head (adversarial)
            out_s = self.adversarial_network(endpoints, x2)
            return loss, out_s
        else:
            # main head (predictive)

            if target is None:
                out = self.predictive_network(x1)
            else:
                out = self.predictive_network(x1, target)
            return out

    def adversarial_network(self, endpoints, x_s):

        # Convolution layers
        endpoints_s = self.backbone.extract_endpoints(x_s)

        x6_s = self.rever1_6(endpoints_s['reduction_6']).mean(dim=2).mean(dim=2)
        x12_s = self.rever1_12(endpoints_s['reduction_6']).std(dim=2).std(dim=2)

        x6_p = self.rever2_6(endpoints_s['reduction_6']).mean(dim=2).mean(dim=2)
        x12_p = self.rever2_12(endpoints_s['reduction_6']).std(dim=2).std(dim=2)

        x = torch.cat([x6_s, x12_s, x6_p, x12_p,], dim=1,)

        x = torch.relu(self.adv_fc1(x))
        # x = torch.relu(self.adv_fc2(x))
        # x = torch.relu(self.adv_fc3(x))
        x = torch.sigmoid(self.adv_fc4(x))

        return x

    def predictive_network(self, inputs, target=None):

        if target == None:
            return self.frcnn(inputs)
        else:
            # Convolution layers
            endpoints = self.backbone.extract_endpoints(torch.stack(inputs, dim=0))
            return self.frcnn(inputs, target), endpoints

    def build_adv_model(self, device):

        dummy = torch.rand(1, 3, 96, 96).to(device)
        endpoints = self.backbone.extract_endpoints(dummy)

        n_filt = endpoints['reduction_6'].shape[1]

        self.adv_fc1 = nn.Linear(n_filt * 4, 300).to(device)
        self.adv_fc2 = nn.Linear(300, 300).to(device)
        # self.adv_fc3 = nn.Linear(300, 300)
        self.adv_fc4 = nn.Linear(300, 1).to(device)

        # gradient reversal layer
        self.rever1_6 = RevGrad().to(device)
        self.rever1_12 = RevGrad().to(device)

        self.rever2_6 = RevGrad().to(device)
        self.rever2_12 = RevGrad().to(device)

        return True
