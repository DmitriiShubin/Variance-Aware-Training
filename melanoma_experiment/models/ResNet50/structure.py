import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet50(nn.Module):
    def __init__(self, hparams, n_classes):
        super(ResNet50, self).__init__()

        self.resnet = resnet18(
            pretrained=False,
            num_classes=1000,
        )

        self.hparams = hparams['model']
        # adversarial deep net layers
        self.out = nn.Linear(1000, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        # TODO
        x = torch.sigmoid(self.out(x))
        return x
