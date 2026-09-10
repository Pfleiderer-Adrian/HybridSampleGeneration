"""Conversion of integer label masks to foreground one-hot tensors."""

import torch
import torch.nn.functional as F


def to_one_hot_3D(mask: torch.Tensor, num_anomaly_classes: int) -> torch.Tensor:
    """Convert a 3D integer mask to ``(B, C, D, H, W)`` foreground channels."""
    if mask.ndim == 5 and mask.shape[1] > 1:
        return mask.float()
    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError(
            f"Expected mask shape (B, D, H, W) after cleanup, got: {mask.shape}."
        )
    mask_oh = F.one_hot(mask.long(), num_classes=num_anomaly_classes + 1)
    return mask_oh[..., 1:].permute(0, 4, 1, 2, 3).float()


def to_one_hot_2D(mask: torch.Tensor, num_anomaly_classes: int) -> torch.Tensor:
    """Convert a 2D integer mask to ``(B, C, H, W)`` foreground channels."""
    if mask.ndim == 4 and mask.shape[1] == num_anomaly_classes:
        return mask.float()
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask shape (B, H, W) after cleanup, got: {mask.shape}."
        )
    mask_oh = F.one_hot(mask.long(), num_classes=num_anomaly_classes + 1)
    return mask_oh[..., 1:].permute(0, 3, 1, 2).float()


__all__ = ["to_one_hot_2D", "to_one_hot_3D"]
