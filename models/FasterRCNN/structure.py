import torch
import torch.nn as nn
import numpy as np
from efficientnet_pytorch import EfficientNet
import torchvision
from torchvision.models.detection import FasterRCNN as FasterRCNN_torchvision
from torchvision.models.detection.rpn import AnchorGenerator

class FasterRCNN(nn.Module):
    def __init__(self, hparams):
        super(FasterRCNN, self).__init__()

        self.hparams = hparams
        self.backbone = EfficientNet.from_pretrained(self.hparams['pre_trained_model'])

        dummy = torch.rand(1, 3, 96, 96)
        endpoints = self.backbone.extract_endpoints(dummy)
        n_filt = endpoints['reduction_6'].shape[1]
        self.backbone.out_channels = n_filt

        self.anchor_generator  = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
        self.frcnn = FasterRCNN_torchvision(self.backbone,
                   num_classes=2,
                   rpn_anchor_generator=self.anchor_generator,
                   box_roi_pool=self.roi_pooler)

    def forward(self, inputs):
        return  self.frcnn(inputs)

