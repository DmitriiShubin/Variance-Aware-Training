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

        # adversarial deep net layers
        self.adv_fc1 = nn.Linear(1000, 1)

    def forward(self, x):
        x_s,x = x

        out, x = self.predictive_network(x)
        # TODO
        out_s = self.adversarial_network(x_s,x)

        return out, out_s

    def predictive_network(self,x):
        x = self.resnet(x)
        out = torch.sigmoid(self.out(x))
        return out,x

    def adversarial_network(self,x_s,x):
        x_s = self.resnet(x_s)

        x = torch.stack([x, x_s], dim=1)

        x = torch.mean(x, dim=3)  # global average pooling only bottleneck of unet
        x = torch.mean(x, dim=3)
        x = torch.mean(x, dim=1)

        x = torch.sigmoid(self.adv_fc1(x))

        return x
