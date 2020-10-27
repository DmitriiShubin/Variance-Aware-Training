import torch
import torch.nn as nn
from loss_functions import AngularPenaltySMLoss
from torchvision.models import resnet50

class ResNet50(nn.Module):
    def __init__(self, hparams, n_channels, n_classes, bilinear=True):
        super(ResNet50, self).__init__()


        self.resnet = resnet50(pretrained=False,
                                num_classes=1000,
                                )

        self.hparams = hparams['model']
        self.n_channels = n_channels
        # adversarial deep net layers
        self.out = nn.Linear(1000, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        #TODO
        x = torch.softmax(self.out(x),dim=1)
        return x