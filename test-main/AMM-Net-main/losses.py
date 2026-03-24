import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_accuracy(y_pred, input_label, bins=10):
    rate_scale = torch.tensor([float(i + 1) for i in range(bins)], device=y_pred.device)
    threshold = float(bins / 2)
    pred_score = torch.sum(y_pred * rate_scale, dim=-1)
    true_score = torch.sum(input_label * rate_scale, dim=-1)
    diff = (((pred_score - threshold) * (true_score - threshold)) >= 0)
    acc = torch.sum(diff.float()) / pred_score.numel()
    return acc


def emd_dis(x, y_true, dist_r=1):
    cdf_x = torch.cumsum(x, dim=-1)
    cdf_ytrue = torch.cumsum(y_true, dim=-1)
    if dist_r == 2:
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
    else:
        samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
    loss = torch.mean(samplewise_emd)
    return loss


def cal_metrics(output, target, bins=10):
    output = np.concatenate(output)
    target = np.concatenate(target)
    score_pred = np.dot(output, np.arange(1, bins + 1))
    score_label = np.dot(target, np.arange(1, bins + 1))
    diff = (((score_pred - float(bins / 2)) * (score_label - float(bins / 2))) >= 0)
    acc_cls = np.sum(diff) / len(score_pred) * 100
    return [score_pred, score_label, acc_cls, output, target]


class emd_loss(nn.Module):
    def __init__(self, dist_r=2, use_l1loss=False, l1loss_coef=0.0):
        super().__init__()
        self.dist_r = dist_r
        self.use_l1loss = use_l1loss
        self.l1loss_coef = l1loss_coef

    def check_type_forward(self, in_types):
        assert len(in_types) == 2
        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0

    def forward(self, x, y_true):
        self.check_type_forward((x, y_true))

        if y_true.size()[1] == 5:
            coff = 1.0 - torch.sum(y_true.pow(2), dim=-1) + 0.2
        else:
            coff = 1.0 - torch.sum(y_true.pow(2), dim=-1) + 0.1

        cdf_x = torch.cumsum(x, dim=-1)
        cdf_ytrue = torch.cumsum(y_true, dim=-1)

        if self.dist_r == 2:
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
        else:
            samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)

        samplewise_emd = samplewise_emd.mul(coff)
        loss = torch.mean(samplewise_emd)

        if self.use_l1loss:
            rate_scale = torch.tensor([float(i + 1) for i in range(x.size()[1])], device=x.device)
            x_mean = torch.sum(x * rate_scale, dim=-1)
            y_true_mean = torch.sum(y_true * rate_scale, dim=-1)

            # Normalize the mean score from [1, bins] to [0, 1] before computing the weight.
            bins = float(x.size()[1])
            y_true_mean_norm = (y_true_mean - 1.0) / max(bins - 1.0, 1.0)
            l1loss_coef = 1.0 - torch.abs(y_true_mean_norm - 0.5)
            l1 = (x_mean - y_true_mean).pow(2)
            l1loss = torch.mean(l1.mul(l1loss_coef))
            if self.l1loss_coef:
                l1loss = l1loss * self.l1loss_coef
            loss += l1loss

        return loss


def pairwise_rank_loss(pred_mean, gt_mean, margin=0.2):
    """
    Batch-wise pairwise margin ranking loss.
    For all pairs (i,j) where gt_mean[i] > gt_mean[j],
    penalise if pred_mean[i] - pred_mean[j] < margin.

    Args:
        pred_mean: (B,) predicted mean scores
        gt_mean:   (B,) ground-truth mean scores
        margin:    minimum required score gap
    Returns:
        scalar loss
    """
    diff_pred = pred_mean.unsqueeze(1) - pred_mean.unsqueeze(0)   # (B, B)
    diff_gt   = gt_mean.unsqueeze(1)   - gt_mean.unsqueeze(0)     # (B, B)
    mask      = (diff_gt > 0).float()                              # pairs where i > j
    rank_loss = F.relu(margin - diff_pred) * mask
    n_pairs   = mask.sum().clamp(min=1)
    return rank_loss.sum() / n_pairs


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)