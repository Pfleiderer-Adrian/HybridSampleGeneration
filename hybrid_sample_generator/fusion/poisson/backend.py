"""Gradient-domain Poisson anomaly fusion backend."""

from __future__ import annotations

import warnings
from dataclasses import replace

from hybrid_sample_generator.fusion.interfaces import FusionOutput
from hybrid_sample_generator.fusion.poisson.configuration import Config
from hybrid_sample_generator.fusion.poisson.spatial import fuse_spatial_poisson
from hybrid_sample_generator.imaging.roi import crop_spatial_clip, dynamic_roi_size


class PoissonFusionBackend:
    """Poisson blending backend for channel-first 2D and 3D samples."""

    def __init__(self, fusion_params: Config | None = None, **kwargs) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown PoissonFusionBackend parameters: {unknown}")
        if fusion_params is not None and not isinstance(fusion_params, Config):
            raise TypeError(f"fusion_params must be {Config.__module__}.Config.")
        self.params = Config() if fusion_params is None else replace(fusion_params)
        self.params.validate()
        self._warned_about_3d = False

    def warmup(self, shape, device=None, dtype=None, config=None):
        self.params.validate()
        return self

    def _warn_for_3d(self, mask_voxels: int, channels: int) -> None:
        if self._warned_about_3d:
            return
        warnings.warn(
            "3D Poisson blending solves a sparse linear system for every channel "
            "and may require substantially more computation time and memory than "
            f"classical alpha blending (mask voxels: {mask_voxels}, "
            f"channels: {channels}).",
            RuntimeWarning,
            stacklevel=3,
        )
        self._warned_about_3d = True

    def fuse(
        self,
        sample: dict,
        control_img,
        position,
        *,
        extraction_config=None,
    ) -> FusionOutput:
        self.params.validate()
        if extraction_config is None:
            raise ValueError(
                "PoissonFusionBackend requires extraction_config for ROI construction."
            )
        if control_img.ndim not in (3, 4):
            raise ValueError(
                f"Unexpected shape: {control_img.shape}, supported shapes are "
                "(C,H,W) and (C,D,H,W)."
            )

        spatial_dims = control_img.ndim - 1
        warning_callback = self._warn_for_3d if spatial_dims == 3 else None
        return fuse_spatial_poisson(
            control_img,
            sample["synth_anomaly"],
            sample["anomaly_meta"],
            position,
            sample["tgt_mask"],
            extraction_config,
            params=self.params,
            spatial_ndim=spatial_dims,
            crop_roi=crop_spatial_clip,
            dynamic_roi_size=dynamic_roi_size,
            anomaly_roi=sample["anomaly_roi"],
            anomaly_roi_mask=sample["anomaly_roi_mask"],
            performance_warning=warning_callback,
        )


__all__ = ["PoissonFusionBackend"]
