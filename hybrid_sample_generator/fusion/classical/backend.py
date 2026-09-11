"""Classical intensity-based anomaly fusion backend."""

from __future__ import annotations

from dataclasses import replace

from hybrid_sample_generator.fusion.classical.alpha import (
    get_alpha_mask_2d,
    get_alpha_mask_3d,
)
from hybrid_sample_generator.fusion.classical.configuration import Config
from hybrid_sample_generator.fusion.classical.spatial import fuse_spatial
from hybrid_sample_generator.fusion.interfaces import FusionOutput
from hybrid_sample_generator.imaging.roi import (
    crop_cube_clip,
    crop_square_clip,
    dynamic_roi_size,
)


class ClassicalFusionBackend:
    """
    Classical alpha-blending fusion backend for 2D and 3D samples.

    High-level steps:
      1) Remove anomaly background via the target mask.
      2) Trim anomaly padding to the foreground bounding box.
      3) Restore anomaly size using the inverse extraction scale.
      4) Compute the insertion box from the normalized position.
      5) Match anomaly intensity to the local control region.
      6) Create an edge-aware alpha mask.
      7) Alpha-blend anomaly and control region.
      8) Create the final segmentation and debug ROI outputs.
    """

    def __init__(self, fusion_params: Config | None = None, **kwargs) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown ClassicalFusionBackend parameters: {unknown}")
        if fusion_params is not None and not isinstance(fusion_params, Config):
            raise TypeError(f"fusion_params must be {Config.__module__}.Config.")
        self.params = Config() if fusion_params is None else replace(fusion_params)
        self.params.validate()

    def warmup(self, shape, device=None, dtype=None, config=None):
        self.params.validate()
        return self

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
                "ClassicalFusionBackend requires extraction_config for ROI construction."
            )

        control = control_img
        anomaly = sample["synth_anomaly"]
        anomaly_meta = sample["anomaly_meta"]
        target_mask = sample["tgt_mask"]
        anomaly_roi = sample["anomaly_roi"]
        anomaly_roi_mask = sample["anomaly_roi_mask"]

        if control.ndim == 3:
            return fuse_spatial(
                control,
                anomaly,
                anomaly_meta,
                position,
                target_mask,
                extraction_config,
                params=self.params,
                spatial_ndim=2,
                crop_roi=crop_square_clip,
                dynamic_roi_size=dynamic_roi_size,
                alpha_builder=get_alpha_mask_2d,
                anomaly_roi=anomaly_roi,
                anomaly_roi_mask=anomaly_roi_mask,
            )
        if control.ndim == 4:
            return fuse_spatial(
                control,
                anomaly,
                anomaly_meta,
                position,
                target_mask,
                extraction_config,
                params=self.params,
                spatial_ndim=3,
                crop_roi=crop_cube_clip,
                dynamic_roi_size=dynamic_roi_size,
                alpha_builder=get_alpha_mask_3d,
                anomaly_roi=anomaly_roi,
                anomaly_roi_mask=anomaly_roi_mask,
            )
        raise ValueError(
            f"Unexpected shape: {control.shape}, Supported: (C, H, W) or "
            "(C, D, H, W)"
        )
