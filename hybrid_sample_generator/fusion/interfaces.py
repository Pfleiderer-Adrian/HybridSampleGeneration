from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import scipy.ndimage as ndi


@dataclass
class FusionOutput:
    """Return value for one fusion operation."""

    image: np.ndarray
    segmentation: np.ndarray
    roi: np.ndarray | None = None
    roi_mask: np.ndarray | None = None
    metrics: dict[str, Any] | None = None


def keep_control_background_after_fusion(
    fused_image: np.ndarray,
    fused_segmentation: np.ndarray,
    control_image: np.ndarray,
    background_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore original control intensities in background regions after fusion."""
    if fused_image.shape != control_image.shape:
        raise ValueError(
            f"fused_image shape {fused_image.shape} does not match control_image shape {control_image.shape}"
        )
    if fused_segmentation.shape != control_image.shape:
        raise ValueError(
            f"fused_segmentation shape {fused_segmentation.shape} does not match control_image shape {control_image.shape}"
        )

    background_mask = np.asarray(background_mask, dtype=bool)
    if background_mask.shape != control_image.shape[1:]:
        raise ValueError(
            f"background_mask shape {background_mask.shape} does not match control spatial shape {control_image.shape[1:]}"
        )
    image = fused_image.copy()
    segmentation = fused_segmentation.copy()
    image[(slice(None), *np.where(background_mask))] = control_image[(slice(None), *np.where(background_mask))]
    segmentation[(slice(None), *np.where(background_mask))] = 0
    return image, segmentation


def control_background_mask(
    control_image: np.ndarray,
    bg_value,
    relative_bg_threshold: float | None = None,
    exterior_only: bool = True,
) -> np.ndarray:
    """Return spatial pixels/voxels that belong to the control background.

    The background cutoff is computed per channel. If bg_value is None, the
    low reference value is estimated from the control sample border. Otherwise
    bg_value is used as the low reference value. relative_bg_threshold expands
    that value by a relative fraction of the channel range.

    control_image is expected to be channel-first, e.g. (C, H, W) or
    (C, D, H, W).
    """
    control_image = np.asarray(control_image)
    if control_image.ndim < 2:
        raise ValueError(f"Expected channel-first image, got shape {control_image.shape}")

    threshold = 0.0 if relative_bg_threshold is None else float(relative_bg_threshold)
    if threshold < 0.0:
        raise ValueError(f"relative_bg_threshold must be >= 0, got {relative_bg_threshold}.")

    border = _border_mask(control_image.shape[1:]) if bg_value is None else None
    cutoffs = []
    for channel in range(control_image.shape[0]):
        channel_values = control_image[channel]
        finite_channel = channel_values[np.isfinite(channel_values)]
        if bg_value is None:
            border_values = channel_values[border]
            border_values = border_values[np.isfinite(border_values)]
            source_values = border_values if border_values.size else finite_channel
            if source_values.size == 0:
                cutoffs.append(0.0)
                continue
            low = _robust_low_background_value(source_values)
        else:
            low = float(bg_value)

        high_values = finite_channel if finite_channel.size else np.asarray([low], dtype=np.float32)
        high = float(np.percentile(high_values, 99.5))
        cutoffs.append(low + threshold * max(high - low, 0.0))

    cutoff_shape = (len(cutoffs),) + (1,) * (control_image.ndim - 1)
    per_channel = control_image <= np.asarray(cutoffs, dtype=np.float32).reshape(cutoff_shape)

    mask = np.all(per_channel, axis=0)
    return _exterior_connected_mask(mask) if exterior_only else mask


def _border_mask(shape: tuple[int, ...]) -> np.ndarray:
    border = np.zeros(shape, dtype=bool)
    for axis in range(len(shape)):
        low = [slice(None)] * len(shape)
        high = [slice(None)] * len(shape)
        low[axis] = 0
        high[axis] = -1
        border[tuple(low)] = True
        border[tuple(high)] = True
    return border


def _robust_low_background_value(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0

    min_value = float(np.min(finite))
    min_count = int(np.count_nonzero(np.isclose(finite, min_value, rtol=0.0, atol=1e-6)))
    if min_count >= max(10, int(0.005 * finite.size)):  # at least 10 pixels or 0.5% of the image, whichever is larger
        return min_value

    return float(np.percentile(finite, 0.5))    # fallback to 0.5th percentile if the minimum is not robust


def _exterior_connected_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only background components connected to the spatial array border."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 3:
        return np.stack([_exterior_connected_mask(slice_mask) for slice_mask in mask], axis=0)
    if not np.any(mask):
        return mask

    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    border = _border_mask(mask.shape)
    border_labels = np.unique(labels[border & mask])
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, border_labels)


@runtime_checkable
class FusionBackend(Protocol):
    """Capability interface consumed by HybridDataGenerator for final sample fusion."""

    def warmup(self, shape, device=None, dtype=None, config=None):
        ...

    def load_checkpoint(self, path: str, **kwargs) -> None:
        ...

    def train_model(
        self,
        sample_dataloader,
        *,
        epochs: int | None = None,
        lr: float | None = None,
        checkpoint_path: str | None = None,
        device=None,
        config=None,
    ) -> dict:
        ...

    def fuse(
        self,
        sample: dict[str, Any],
        control_img: np.ndarray,
        position: Any,
        *,
        extraction_config=None,
    ) -> FusionOutput:
        ...
