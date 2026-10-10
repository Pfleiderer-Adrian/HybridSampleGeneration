"""Classical intensity-based anomaly fusion backend."""

from __future__ import annotations

from dataclasses import replace

from hybrid_sample_generator.fusion.classical.alpha import (
    get_alpha_mask_2d,
    get_alpha_mask_3d,
)
from hybrid_sample_generator.fusion.classical.configuration import Config
from hybrid_sample_generator.fusion.classical.harmonization import ImageHarmonizer
from hybrid_sample_generator.fusion.classical.spatial import fuse_spatial
from hybrid_sample_generator.fusion.interfaces import FusionOutput
from hybrid_sample_generator.imaging.roi import crop_spatial_clip, dynamic_roi_size


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
        self._harmonizer = None

    def warmup(self, shape, device=None, dtype=None, config=None):
        self.params.validate()
        if self.params.image_harmonization is not None:
            self._validate_harmonization_shape(shape)
            self._get_harmonizer(device=device)
        return self

    def _get_harmonizer(self, *, device=None):
        if self.params.image_harmonization is None:
            return None
        if self._harmonizer is None:
            self._harmonizer = ImageHarmonizer(
                self.params.image_harmonization,
                device=device,
            )
        return self._harmonizer

    @staticmethod
    def _validate_harmonization_shape(shape) -> None:
        shape = tuple(shape)
        if len(shape) != 3:
            raise ValueError(
                "PCTNet and LBM image harmonization support only 2D samples "
                f"with shape (C,H,W); got {shape}."
            )
        if shape[0] not in (1, 3):
            raise ValueError(
                "PCTNet and LBM image harmonization require one or three channels; "
                f"got {shape[0]}."
            )

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

        if control.ndim not in (3, 4):
            raise ValueError(
                f"Unexpected shape: {control.shape}, supported shapes are "
                "(C,H,W) and (C,D,H,W)."
            )
        spatial_dims = control.ndim - 1
        alpha_builder = get_alpha_mask_2d if spatial_dims == 2 else get_alpha_mask_3d
        harmonizer = None
        if self.params.image_harmonization is not None:
            self._validate_harmonization_shape(control.shape)
            harmonizer = self._get_harmonizer()
        return fuse_spatial(
            control,
            anomaly,
            anomaly_meta,
            position,
            target_mask,
            extraction_config,
            params=self.params,
            spatial_ndim=spatial_dims,
            crop_roi=crop_spatial_clip,
            dynamic_roi_size=dynamic_roi_size,
            alpha_builder=alpha_builder,
            anomaly_roi=anomaly_roi,
            anomaly_roi_mask=anomaly_roi_mask,
            image_postprocessor=harmonizer,
        )
