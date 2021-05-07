import torch
import torch.nn as nn
import numpy as np
from models.FasterRCNN.EfficientNet import EfficientNet
import torchvision
from torchvision.models.detection import FasterRCNN as FasterRCNN_torchvision
from torchvision.models.detection.rpn import AnchorGenerator


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

        #self.sub_module_b_dict = nn.ModuleDict({'frcnn': self.frcnn})

    def forward(self, inputs, target=None):
        if target == None:
            return self.frcnn(inputs)
        else:
            return self.frcnn(inputs, target)
