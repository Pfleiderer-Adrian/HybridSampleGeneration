"""Differentiable MSE + Gaussian-window SSIM + focal segmentation loss."""

import torch
from torch.nn import functional as F


def focal_loss(logits, mask, gamma=2.0):
    ce = F.cross_entropy(logits, mask[:, 0].long(), reduction="none")
    return ((1 - torch.exp(-ce)).pow(gamma) * ce).mean()


def ssim_loss(prediction, target):
    axis = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    weights = torch.exp(-axis.square() / (2 * 1.5**2))
    weights /= weights.sum()
    window = (weights[:, None] * weights[None, :]).expand(prediction.shape[1], 1, 11, 11)
    def average(x):
        return F.conv2d(x, window, padding=5, groups=x.shape[1])
    a, b = average(prediction), average(target)
    var_a, var_b = average(prediction.square()) - a.square(), average(target.square()) - b.square()
    covariance = average(prediction * target) - a * b
    similarity = ((2*a*b + 0.01**2) * (2*covariance + 0.03**2)) / ((a.square()+b.square()+0.01**2) * (var_a+var_b+0.03**2))
    return 1 - similarity.mean()


def training_loss(reconstruction, logits, target, mask):
    return F.mse_loss(reconstruction, target) + ssim_loss(reconstruction, target) + focal_loss(logits, mask)
