import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    '''
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
    '''

    def __init__(self, batch_size, device,normalize=False, temperature=0.1):
        super(SimclrCriterion, self).__init__()

        self.temperature = temperature
        self.normalize = normalize
        self.device = device

        self.register_buffer('labels', torch.zeros(batch_size * 2).long().to(device))

        self.register_buffer('mask', torch.ones(
            (batch_size, batch_size), dtype=bool).fill_diagonal_(0))

    def forward(self, z_i, z_j):

        if self.normalize:
            z_i_norm = F.normalize(z_i, p=2, dim=-1)
            z_j_norm = F.normalize(z_j, p=2, dim=-1)

        else:
            z_i_norm = z_i
            z_j_norm = z_j

        bsz = z_i_norm.size(0)

        ''' Note: **
        Cosine similarity matrix of all samples in batch:
        a = z_i
        b = z_j
         ____ ____
        | aa | ab |
        |____|____|
        | ba | bb |
        |____|____|
        Postives:
        Diagonals of ab and ba '\'
        Negatives:
        All values that do not lie on leading diagonals of aa, bb, ab, ba.
        '''

        # Cosine similarity between all views
        logits_aa = torch.mm(z_i_norm, z_i_norm.t()) / self.temperature
        logits_bb = torch.mm(z_j_norm, z_j_norm.t()) / self.temperature
        logits_ab = torch.mm(z_i_norm, z_j_norm.t()) / self.temperature
        logits_ba = torch.mm(z_j_norm, z_i_norm.t()) / self.temperature

        mask = torch.ones((z_i.shape[0], z_i.shape[0]), dtype=bool).fill_diagonal_(0).to(self.device)
        labels = torch.ones(z_i.shape[0]* 2).long().to(self.device)

        # Compute Postive Logits
        logits_ab_pos = logits_ab[torch.logical_not(mask)]
        logits_ba_pos = logits_ba[torch.logical_not(mask)]

        # Compute Negative Logits
        logit_aa_neg = logits_aa[mask].reshape(bsz, -1)
        logit_bb_neg = logits_bb[mask].reshape(bsz, -1)
        logit_ab_neg = logits_ab[mask].reshape(bsz, -1)
        logit_ba_neg = logits_ba[mask].reshape(bsz, -1)

        # Postive Logits over all samples
        pos = torch.cat((logits_ab_pos, logits_ba_pos)).unsqueeze(1)

        # Negative Logits over all samples
        neg_a = torch.cat((logit_aa_neg, logit_ab_neg), dim=1)
        neg_b = torch.cat((logit_ba_neg, logit_bb_neg), dim=1)

        neg = torch.cat((neg_a, neg_b), dim=0)

        # Compute cross entropy
        logits = torch.cat((pos, neg), dim=1)

        loss = F.cross_entropy(logits,labels)

        return loss


class SimCLR_2(nn.Module):
    def __init__(self,temperature=1):
        super(SimCLR_2, self).__init__()
        self.temperature = temperature

    def forward(self, out_1, out_2 , eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     out_1_dist = SyncFunction.apply(out_1)
        #     out_2_dist = SyncFunction.apply(out_2)
        # else:
        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

class contrastive_loss(nn.Module):
    def __init__(self, tau=1, normalize=True):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        #x = torch.cat((xi, xj), dim=0)

        is_cuda = xi.is_cuda
        sim_mat = torch.mm(xi, xj.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(xi, dim=1).unsqueeze(1), torch.norm(xj, dim=1).unsqueeze(1).T)
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

        norm_sum = torch.exp(torch.ones(xi.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
        return loss

class local_contrastive_loss(nn.Module):
    def __init__(self, tau=10000, normalize=True):
        super(local_contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        #x = torch.cat((xi, xj), dim=0)

        is_cuda = xi.is_cuda
        sim_mat = torch.mm(xi.T, xj)

        sim_mat = torch.exp(sim_mat / self.tau)

        sum = torch.sum(sim_mat)

        loss = -torch.mean(torch.log(sim_mat / sum))
        return loss
