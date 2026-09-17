"""Conversion of integer label masks to foreground one-hot tensors."""

import torch
import torch.nn.functional as F


def to_one_hot(
    mask: torch.Tensor,
    num_anomaly_classes: int,
    *,
    spatial_dims: int,
) -> torch.Tensor:
    """Convert integer labels to channel-first foreground one-hot masks."""
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
    channel_first_ndim = spatial_dims + 2
    if mask.ndim == channel_first_ndim and mask.shape[1] == num_anomaly_classes:
        return mask.float()
    if mask.ndim == channel_first_ndim and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.ndim == spatial_dims:
        mask = mask.unsqueeze(0)
    if mask.ndim != spatial_dims + 1:
        raise ValueError(
            f"Expected a batch plus {spatial_dims} spatial dimensions after cleanup, "
            f"got: {mask.shape}."
        )
    channel_last = F.one_hot(mask.long(), num_classes=num_anomaly_classes + 1)[..., 1:]
    order = (0, channel_last.ndim - 1, *range(1, channel_last.ndim - 1))
    return channel_last.permute(order).float()


__all__ = ["to_one_hot"]
