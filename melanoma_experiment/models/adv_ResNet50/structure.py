import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, hparams, n_classes):
        super(ResNet50, self).__init__()

        self.resnet = resnet50(
            pretrained=False,
            num_classes=1000,
        )

        self.hparams = hparams['model']
        # adversarial deep net layers
        self.out = nn.Linear(1000, n_classes)

        # adversarial deep net layers
        self.adv_fc1 = nn.Linear(2000, 300)
        self.adv_fc2 = nn.Linear(300, 1)

    def forward(self, x):
        x_s, x = x

        out, x = self.predictive_network(x)
        # TODO
        out_s = self.adversarial_network(x_s, x)

        return out, out_s

    def predictive_network(self, x):
        x = self.resnet(x)
        out = torch.sigmoid(self.out(x))
        return out, x

    def adversarial_network(self, x_s, x):
        x_s = self.resnet(x_s)

        x = torch.cat((x, x_s), dim=1)
        x = torch.relu(self.adv_fc1(x))
        x = torch.sigmoid(self.adv_fc2(x))

        return x
