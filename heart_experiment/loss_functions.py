import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()

        self.smoothing = 1

    def forward(self, y_true, y_pred):
        # y_truef = torch.flatten(y_true)
        # y_predf = torch.flatten(y_pred)
        And = torch.sum(y_true * y_pred, dim=0)
        return -1 * torch.mean(
            (2 * And + self.smoothing)
            / (torch.sum(y_true, dim=0) + torch.sum(y_pred, dim=0) + self.smoothing)
        )


class Jaccard_loss(nn.Module):
    def __init__(self):
        super(Jaccard_loss, self).__init__()

        self.smoothing = 1

    def forward(self, y_true, y_pred):
        Intersection = torch.sum(y_true * y_pred, dim=0)
        Union = torch.sum(y_true + y_pred, dim=0) - Intersection
        return -1 * torch.mean((Intersection+ self.smoothing)/(Union+ self.smoothing))
