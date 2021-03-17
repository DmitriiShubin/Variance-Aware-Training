import torch
import torch.nn as nn
import torch.nn.functional as F


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
            tp
            / (tp + 0.5 * (fp + fn) + self.smoothing)
        )
        return -1 * f1


class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()

        self.smoothing = 1e-5

    def forward(self, y_pred, y_true):
        # y_truef = torch.flatten(y_true)
        # y_predf = torch.flatten(y_pred)
        y_true = y_true[:, 1:] #removing background makes less stable
        y_pred = y_pred[:, 1:]
        tp = torch.sum(y_true * y_pred, dim=0)
        fp = torch.sum(y_pred, dim=0) - tp
        fn = torch.sum(y_true, dim=0) - tp

        f1 = torch.mean(
            (tp + self.smoothing) / (tp + 0.5*(fn + fp) + self.smoothing)
        )

        # f1 = torch.mean(
        #     ((1 + 2 ** 2) * tp + self.smoothing) / ((1 + 2 ** 2) * tp + 2 ** 2 * fn + fp + self.smoothing)
        # )
        return -1 * f1


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


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (
            target.float() * distances
            + (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        )
        return losses.mean() if size_average else losses.sum()


class SimclrCriterion(nn.Module):
    """
    Taken from: https://github.com/google-research/simclr/blob/master/objective.py
    Converted to pytorch, and decomposed for a clearer understanding.
    Args:
        init:
            batch_size (integer): Number of datasamples per batch.
            normalize (bool, optional): Whether to normalise the reprentations.
                (Default: True)
            temperature (float, optional): The temperature parameter of the
                NT_Xent loss. (Default: 1.0)
        forward:
            z_i (Tensor): Reprentation of view 'i'
            z_j (Tensor): Reprentation of view 'j'
    Returns:
        loss (Tensor): NT_Xent loss between z_i and z_j
    """

    def __init__(self, batch_size, normalize=True, temperature=1.0):
        super(SimclrCriterion, self).__init__()

        self.temperature = temperature
        self.normalize = normalize

        self.register_buffer('labels', torch.zeros(batch_size * 2).long())

        self.register_buffer('mask', torch.ones((batch_size, batch_size), dtype=bool).fill_diagonal_(0))

    def forward(self, z1, z2):
        sim11 = self.cossim(z1.unsqueeze(-2), z1.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(z2.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(z1.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)

        loss = F.cross_entropy(raw_scores, targets)

        return loss


class contrastive_loss(nn.Module):
    def __init__(self, tau=1, normalize=True):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
        return loss