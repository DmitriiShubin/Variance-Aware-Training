import torch
import torch.nn as nn


class f1_loss(nn.Module):
    def __init__(self):
        super(f1_loss, self).__init__()

        self.smoothing = 1e-5

    def forward(self, y_pred, y_true):
        # y_pred = y_pred[:,1:]
        # y_true = y_true[:, 1:]

        tp = torch.sum(y_true * y_pred, dim=0)
        fp = torch.sum(y_pred, dim=0) - tp
        fn = torch.sum(y_true, dim=0) - tp

        f1 = torch.mean(
            # ((1 + 2 ** 2) * tp + self.smoothing) / ((1 + 2 ** 2) * tp + 2 ** 2 * fn + fp + self.smoothing)
            tp / (tp + 0.5 * (fp + fn) + self.smoothing)
        )
        return -1 * f1

class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()

        self.smoothing = 1e-5

    def forward(self, y_pred, y_true):
        # y_truef = torch.flatten(y_true)
        # y_predf = torch.flatten(y_pred)
        y_true = y_true[:, 1:]
        y_pred = y_pred[:, 1:]
        # tp = torch.sum(y_true * y_pred, dim=0)
        # fp = torch.sum(y_pred, dim=0) - tp
        # fn = torch.sum(y_true, dim=0) - tp

        # f1 = torch.mean(
        #     ((1 + 2 ** 2) * tp + self.smoothing) / ((1 + 2 ** 2) * tp + 2 ** 2 * fn + fp + self.smoothing)
        # )

        f1 = 2*torch.sum(y_pred*y_true,dim=0)/(torch.sum(y_pred**2,dim=0) + torch.sum(y_true**2,dim=0) + self.smoothing)

        return -1 * torch.mean(f1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        p_t = torch.where(target == 1, x, 1 - x)
        fl = -1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        fl = torch.where(target == 1, fl * self.alpha, fl)
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
