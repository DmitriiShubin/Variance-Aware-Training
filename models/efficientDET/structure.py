import torch
import torch.nn as nn

from models.efficientDET.efficientnet import EfficientNet

from models.efficientDET.bifpn import BIFPN
from models.efficientDET.anchors import Anchors
from models.efficientDET.retinahead import RetinaHead
from models.efficientDET.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
import math
from torchvision.ops import nms
import utils.losses_object_det as losses


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(
            num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(
            num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size,
                               kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)
        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height,
                         self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class EfficientDet(nn.Module):
    """
    original repo: https://github.com/tristandb/EfficientDet-PyTorch/
    """

    def __init__(self, hparams,device):
        super().__init__()
        self.hparams = hparams
        self.device = device
        num_classes = self.hparams['n_classes']
        phi = 4
        w_bifpn = [64, 88, 112, 160, 224, 288, 384, 384]

        self.inplanes = w_bifpn[phi]


        self.backbone = EfficientNet.from_pretrained(self.hparams['pre_trained_model'])
        self.neck = BIFPN(in_channels=self.backbone.get_list_features()[-5:],
                          out_channels=self.hparams['W_bifpn'],
                          stack=self.hparams['D_bifpn'],
                          num_outs=5)
        self.bbox_head = RetinaHead(num_classes=num_classes,
                                    in_channels=self.hparams['W_bifpn'])

        self.focalLoss = losses.FocalLoss(device=self.device)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()

    def forward(self, img_batch, annotations=None,training=False):


        x = self.extract_feat(img_batch)
        outs = self.bbox_head(x)

        classification = torch.cat([out for out in outs[0]], dim=1)
        regression = torch.cat([out for out in outs[1]], dim=1)
        anchors = self.anchors(img_batch)



        if training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)


            return [classification, transformed_anchors]

    def extract_feat(self, img):
        """
            Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x[-5:])
        return x

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()