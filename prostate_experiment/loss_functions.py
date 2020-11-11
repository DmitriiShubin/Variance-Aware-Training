import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()

        self.smoothing = 1e-5

    def forward(self, y_pred, y_true):
        # y_truef = torch.flatten(y_true)
        # y_predf = torch.flatten(y_pred)
        y_true = y_true[:, 1:]
        y_pred = y_pred[:, 1:]
        tp = torch.sum(y_true * y_pred, dim=0)
        fp = torch.sum(y_pred, dim=0) - tp
        fn = torch.sum(y_true, dim=0) - tp

        f1 = torch.mean(((1 + 2 ** 2) * tp + self.smoothing) \
            / ((1 + 2 ** 2) * tp + 2 ** 2 * fn + fp + self.smoothing))
        return -1*f1
        # And = torch.sum(y_true * y_pred,dim=0)
        # return 1 - torch.mean(
        #     (2 * And + self.smoothing)
        #     / (torch.sum(y_true,dim=0) + torch.sum(y_pred,dim=0) + self.smoothing)
        # )


class Jaccard_loss(nn.Module):
    def __init__(self):
        super(Jaccard_loss, self).__init__()

        self.smoothing = 1e-5

    def forward(self, y_pred, y_true):
        # y_true = y_true[:,1]
        # y_pred = y_pred[:, 1]
        Intersection = torch.sum(y_true * y_pred, dim=0)
        Union = torch.sum(y_true, dim=0) + torch.sum(y_pred, dim=0) - Intersection
        loss = torch.mean((Intersection+ self.smoothing)/(Union+ self.smoothing))

        # tp = torch.sum(y_true * y_pred, dim=0)
        # fp = torch.sum((1 - y_true) * y_pred, dim=0)
        # fn = torch.sum(y_true * (1 - y_pred), dim=0)
        # loss = (tp+ self.smooth) / (tp + 1 * (fp + fn + self.smoothing))

        return -1*loss
