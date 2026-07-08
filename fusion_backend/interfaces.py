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
    bg_value,
    background_threshold: float | None = None,
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

    background_mask = control_background_mask(
        control_image,
        bg_value,
        background_threshold,
        spatial=True,
        exterior_only=True,
    )
    image = fused_image.copy()
    segmentation = fused_segmentation.copy()
    image[(slice(None), *np.where(background_mask))] = control_image[(slice(None), *np.where(background_mask))]
    segmentation[(slice(None), *np.where(background_mask))] = 0
    return image, segmentation


def control_background_mask(
    control_image: np.ndarray,
    bg_value,
    background_threshold: float | None = None,
    *,
    spatial: bool = False,
    exterior_only: bool = False,
) -> np.ndarray:
    """Return pixels/voxels that belong to the control background.

    If bg_value is None, estimate a per-channel cutoff from the current
    control sample. The cutoff starts at the low exterior intensity and adds
    background_threshold as a relative fraction of the channel range. Explicit
    numeric bg_value keeps the old absolute-threshold behavior.
    """
    control_image = np.asarray(control_image)
    if bg_value is None:
        cutoffs = _relative_background_cutoffs(control_image, background_threshold)
        per_channel = control_image <= _channel_cutoff_shape(cutoffs, control_image.ndim)
    else:
        threshold = 0.0 if background_threshold is None else float(background_threshold)
        if threshold < 0.0:
            raise ValueError(f"background_threshold must be >= 0, got {background_threshold}.")
        if threshold <= 0:
            per_channel = control_image <= bg_value
        else:
            per_channel = control_image <= bg_value + threshold

    if spatial:
        if per_channel.ndim < 2:
            raise ValueError(f"Expected channel-first image, got shape {control_image.shape}")
        mask = np.all(per_channel, axis=0)
        return _exterior_connected_mask(mask) if exterior_only else mask
    if exterior_only:
        raise ValueError("exterior_only=True requires spatial=True.")
    return per_channel



def _channel_cutoff_shape(cutoffs: np.ndarray, ndim: int) -> np.ndarray:
    if ndim <= 1:
        return cutoffs
    return cutoffs.reshape((cutoffs.size,) + (1,) * (ndim - 1))


def _relative_background_cutoffs(control_image: np.ndarray, background_threshold: float | None = None) -> np.ndarray:
    if control_image.ndim < 2:
        values = control_image[np.isfinite(control_image)]
        if values.size == 0:
            return np.asarray(0.0, dtype=np.float32)
        threshold_rel = _relative_background_threshold(background_threshold)
        low = _robust_low_background_value(values)
        high = float(np.percentile(values, 99.5))
        return np.asarray(low + threshold_rel * max(high - low, 0.0), dtype=np.float32)

    threshold_rel = _relative_background_threshold(background_threshold)
    spatial_shape = control_image.shape[1:]
    border = _spatial_border_mask(spatial_shape)
    cutoffs = []
    for channel in range(control_image.shape[0]):
        channel_values = control_image[channel]
        finite_channel = channel_values[np.isfinite(channel_values)]
        border_values = channel_values[border]
        border_values = border_values[np.isfinite(border_values)]
        source_values = border_values if border_values.size else finite_channel
        if source_values.size == 0:
            cutoffs.append(0.0)
            continue

        low = _robust_low_background_value(source_values)
        high_values = finite_channel if finite_channel.size else source_values
        high = float(np.percentile(high_values, 99.5))
        cutoffs.append(low + threshold_rel * max(high - low, 0.0))

    return np.asarray(cutoffs, dtype=np.float32)


def _relative_background_threshold(background_threshold: float | None) -> float:
    threshold_rel = 0.0 if background_threshold is None else float(background_threshold)
    if threshold_rel < 0.0:
        raise ValueError(f"background_threshold must be >= 0, got {background_threshold}.")
    return threshold_rel


def _robust_low_background_value(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0

    min_value = float(np.min(finite))
    min_count = int(np.count_nonzero(np.isclose(finite, min_value, rtol=0.0, atol=1e-6)))
    if min_count >= max(8, int(0.005 * finite.size)):
        return min_value

    return float(np.percentile(finite, 0.5))


def _spatial_border_mask(spatial_shape: tuple[int, ...]) -> np.ndarray:
    border = np.zeros(spatial_shape, dtype=bool)
    for axis in range(len(spatial_shape)):
        low = [slice(None)] * len(spatial_shape)
        high = [slice(None)] * len(spatial_shape)
        low[axis] = 0
        high[axis] = -1
        border[tuple(low)] = True
        border[tuple(high)] = True
    return border


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

    border = np.zeros_like(mask, dtype=bool)
    for axis in range(mask.ndim):
        low = [slice(None)] * mask.ndim
        high = [slice(None)] * mask.ndim
        low[axis] = 0
        high[axis] = -1
        border[tuple(low)] = True
        border[tuple(high)] = True

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
        config=None,
    ) -> FusionOutput:
        ...
