import torch
import torch.nn as nn
import numpy as np
from models.FasterRCNN_pre_trained.EfficientNet import EfficientNet
import torchvision
from torchvision.models.detection import FasterRCNN as FasterRCNN_torchvision
from torchvision.models.detection.rpn import AnchorGenerator

from models.encoder_contrastive_classification.structure import EfficientNet as effnet_contrastive
from models.encoder_patch_classification.structure import EfficientNet as effnet_patch
from models.encoder_rotation_classification.structure import EfficientNet as effnet_rotation
import yaml


class FasterRCNN(nn.Module):
    def __init__(self, hparams, device):
        super(FasterRCNN, self).__init__()

        self.hparams = hparams
        self.backbone = EfficientNet.from_pretrained(self.hparams['pre_trained_model'])

        self.anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        ).to(device)
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        ).to(device)

        # self.sub_module_b_dict = nn.ModuleDict({'frcnn': self.frcnn})

    def forward(self, inputs, target=None):
        if target == None:
            return self.frcnn(inputs)
        else:
            return self.frcnn(inputs, target)

    def load_self_supervised_model(
        self, type_pretrain: str, pre_trained_model: str, pre_trained_model_ssl: str, device
    ):

        hparams = yaml.load(open(pre_trained_model_ssl + '_hparams.yml'), Loader=yaml.FullLoader)

        if type_pretrain == 'contrastive':
            self.encoder = effnet_contrastive.from_pretrained(pre_trained_model).to(device)
            self.encoder.build_projection_network(hparams['model']['emb_dim'], device=device)
        elif type_pretrain == 'patch':
            self.encoder = effnet_patch.from_pretrained(pre_trained_model).to(device)
            self.encoder.build_projection_network(device=device)
        elif type_pretrain == 'rotation':
            self.encoder = effnet_rotation.from_pretrained(pre_trained_model, num_classes=4).to(device)

        self.encoder.load_state_dict(torch.load(pre_trained_model_ssl + '.pt'))

        dummy = torch.rand(1, 3, 96, 96).to(device)
        endpoints = self.encoder.extract_endpoints(dummy)
        n_filt = endpoints['reduction_6'].shape[1]
        self.encoder.out_channels = n_filt

        dummy = dummy.cpu().detach()
        endpoints = {k: v.cpu().detach() for k, v in endpoints.items()}

        self.frcnn = FasterRCNN_torchvision(
            self.encoder,
            num_classes=self.hparams['n_classes'],
            rpn_anchor_generator=self.anchor_generator,
            box_roi_pool=self.roi_pooler,
        ).to(device)

        return True
