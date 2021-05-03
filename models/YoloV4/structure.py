import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from models.YoloV4.FPN import FPN
import numpy as np
from metrics.metrics import get_iou

from models.YoloV4.Prediction_head import PredictionHead


class YoloV4(nn.Module):
    """
    https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/4ccef0ec8fe984e059378813e33b3740929e0c19
    """

    def __init__(self, hparams, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        # backbone
        self.backbone = EfficientNet.from_pretrained(hparams['pre_trained_model'])

        self.backbone.eval()
        dummy = torch.rand(1, 3, 512, 512)
        endpoints = self.backbone.extract_endpoints(dummy)
        self.backbone.train()

        # neck
        self.neek = FPN(
            endpoints['reduction_3'].shape[1],
            endpoints['reduction_4'].shape[1],
            endpoints['reduction_5'].shape[1],
        )

        self.head = PredictionHead(
            endpoints['reduction_3'].shape[1], n_classes=hparams['n_classes'], image_size=512
        )

    def forward(self, input, train=False):

        endpoints = self.backbone.extract_endpoints(input)

        neek1, neek2, neek3 = self.neek(
            endpoints['reduction_3'], endpoints['reduction_4'], endpoints['reduction_5'],
        )

        output = self.head(neek1, neek2, neek3, train=train)
        return output
