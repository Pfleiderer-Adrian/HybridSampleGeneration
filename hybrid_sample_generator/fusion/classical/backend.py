"""Classical intensity-based anomaly fusion backend."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import binary_dilation, zoom

from hybrid_sample_generator.fusion.classical.alpha import (
    get_alpha_mask_2d as _get_alpha_mask_2d,
    get_alpha_mask_3d as _get_alpha_mask_3d,
)
from hybrid_sample_generator.fusion.classical.configuration import Config
from hybrid_sample_generator.fusion.classical.intensity import (
    anomaly_context_relations as _anomaly_context_relations,
    infer_output_intensity_bounds as _infer_output_intensity_bounds,
    normalize_anomaly_to_context as _normalize_anomaly_to_context,
)
from hybrid_sample_generator.fusion.interfaces import FusionOutput, control_background_mask, keep_control_background_after_fusion
from hybrid_sample_generator.imaging.roi import (
    crop_cube_clip,
    crop_square_clip,
    dynamic_roi_size,
)
from hybrid_sample_generator.imaging.masks.interpolation import interpolate_masked_regions
from hybrid_sample_generator.fusion.preprocessing import (
    denormalize_anomaly as _denormalize_anomaly,
    inverse_extraction_scale as _inverse_extraction_scale,
    spatial_label_mask as _spatial_label_mask,
    validate_position as _validate_position,
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
            raise ValueError("ClassicalFusionBackend requires extraction_config for ROI construction.")

        control = control_img
        anomaly = sample["synth_anomaly"]
        anomaly_meta = sample["anomaly_meta"]
        target_mask = sample["tgt_mask"]
        anomaly_roi = sample["anomaly_roi"]
        anomaly_roi_mask = sample["anomaly_roi_mask"]

        if control.ndim == 3:
            return self._fuse_spatial(
                control,
                anomaly,
                anomaly_meta,
                position,
                target_mask,
                extraction_config,
                spatial_ndim=2,
                crop_roi=crop_square_clip,
                dynamic_roi_size=dynamic_roi_size,
                alpha_builder=_get_alpha_mask_2d,
                anomaly_roi=anomaly_roi,
                anomaly_roi_mask=anomaly_roi_mask,
            )
        if control.ndim == 4:
            return self._fuse_spatial(
                control,
                anomaly,
                anomaly_meta,
                position,
                target_mask,
                extraction_config,
                spatial_ndim=3,
                crop_roi=crop_cube_clip,
                dynamic_roi_size=dynamic_roi_size,
                alpha_builder=_get_alpha_mask_3d,
                anomaly_roi=anomaly_roi,
                anomaly_roi_mask=anomaly_roi_mask,
            )
        raise ValueError(f"Unexpected shape: {control.shape}, Supported: (C, H, W) or (C, D, H, W)")

    def _fuse_spatial(
        self,
        control,
        anomaly,
        anomaly_meta,
        position,
        target_mask,
        extraction_config,
        *,
        spatial_ndim: int,
        crop_roi,
        dynamic_roi_size,
        alpha_builder,
        anomaly_roi=None,
        anomaly_roi_mask=None,
    ) -> FusionOutput:
        """
        Shared implementation for 2D and 3D classical fusion.

        The old `fusion2d` and `fusion3d` code paths only differed in spatial
        dimensionality, ROI crop function, and alpha mask construction. Keeping
        the common flow here prevents the two variants from drifting apart.
        """
        if anomaly_meta is None:
            raise ValueError("anomaly_meta must be provided (needs at least 'scale_factor').")

        scale_factor = anomaly_meta.get("scale_factor")
        if scale_factor is None:
            raise ValueError("anomaly_meta is missing required key 'scale_factor'.")
        if target_mask is None:
            raise ValueError("ClassicalFusionBackend requires target_mask. Create or load tgt_mask before fusion.")

        # Ensure float32 for arithmetic stability without copying when possible.
        ctrl = control.astype(np.float32, copy=False)
        anom = anomaly.astype(np.float32, copy=False)
        anom = _denormalize_anomaly(anom, anomaly_meta)
        target_mask = np.asarray(target_mask)

        # Validate dimensionality. Inputs are expected to be channel-first:
        # 2D: (C, H, W), 3D: (C, D, H, W).
        expected_ndim = spatial_ndim + 1
        if ctrl.ndim != expected_ndim or anom.ndim != expected_ndim:
            expected = "(C,H,W)" if spatial_ndim == 2 else "(C,D,H,W)"
            raise ValueError(f"Both images must be {expected}. Got ctrl={ctrl.shape}, anomaly={anom.shape}")
        if target_mask.ndim not in (spatial_ndim, expected_ndim):
            raise ValueError(
                f"target_mask must have {spatial_ndim} or {expected_ndim} dims. Got {target_mask.shape}"
            )
        if target_mask.shape[-spatial_ndim:] != anom.shape[-spatial_ndim:]:
            raise ValueError(
                f"target mask spatial shape {target_mask.shape[-spatial_ndim:]} "
                f"does not match anomaly {anom.shape[-spatial_ndim:]}"
            )

        channels = ctrl.shape[0]
        ctrl_spatial = np.array(ctrl.shape[1:], dtype=int)

        # ------------------------------------------------------------
        # 1) Remove anomaly background by pushing pixels outside the
        #    target mask to the anomaly minimum. This increases contrast
        #    between foreground and background and stabilizes mask creation.
        # ------------------------------------------------------------
        foreground = _spatial_label_mask(target_mask, spatial_ndim) > 0
        background_values = anom[:, ~foreground]
        finite_background = background_values[np.isfinite(background_values)]
        bg_min = (
            float(np.min(finite_background))
            if finite_background.size
            else float(np.nanmin(anom))
        )
        anom = np.where(foreground[None, ...], anom, bg_min)

        # ------------------------------------------------------------
        # 2) Trim spatial padding by cropping to the foreground bounding
        #    box. The same crop is used for all channels.
        # ------------------------------------------------------------
        # Foreground mask over spatial dims: target mask defines the
        # intended label footprint.
        if np.any(foreground):
            coords = np.where(foreground)
            crop_slices = tuple(slice(axis.min(), axis.max() + 1) for axis in coords)
            anom = anom[(slice(None), *crop_slices)]
            if target_mask.ndim == expected_ndim:
                target_mask = target_mask[(slice(None), *crop_slices)]
            else:
                target_mask = target_mask[crop_slices]
        # If there is no foreground, anomaly and target mask remain unchanged.

        # ------------------------------------------------------------
        # 3) Restore anomaly footprint from extraction scale while keeping
        #    the channel axis unchanged.
        # ------------------------------------------------------------
        scale = _inverse_extraction_scale(scale_factor, ndim=spatial_ndim)
        spatial_target_mask = _spatial_label_mask(target_mask, spatial_ndim)
        anom = interpolate_masked_regions(
            anom,
            spatial_target_mask > 0,
            warp=lambda spatial: zoom(spatial, scale, order=1),
            nearest_warp=lambda spatial: zoom(spatial, scale, order=0),
            interpolate_background=False,
            background_fill=bg_min,
        )
        target_mask = zoom(spatial_target_mask, scale, order=0).astype(
            np.uint8, copy=False
        )

        # ------------------------------------------------------------
        # 4) Compute insertion offset from normalized position.
        # ------------------------------------------------------------
        position = _validate_position(position, spatial_ndim)
        anom_spatial = np.array(anom.shape[1:], dtype=int)

        # Offset is chosen so the anomaly center is placed at
        # position * control_size.
        offset = np.array(
            [
                round(ctrl_spatial[axis] * position[axis] - anom_spatial[axis] / 2)
                for axis in range(spatial_ndim)
            ],
            dtype=int,
        )
        offset_end = offset + anom_spatial

        # ------------------------------------------------------------
        # 5) Clamp insertion box to control bounds.
        # ------------------------------------------------------------
        for axis, (start, end, limit) in enumerate(zip(offset, offset_end, ctrl_spatial)):
            # If end exceeds bounds, shift the box back into the control image.
            if end > limit:
                shift = end - limit
                offset[axis] -= shift
                offset_end[axis] -= shift
            # If start is negative, shift the box forward into the control image.
            if offset[axis] < 0:
                shift = -offset[axis]
                offset[axis] += shift
                offset_end[axis] += shift

        insert_slices = tuple(slice(int(start), int(end)) for start, end in zip(offset, offset_end))

        # ------------------------------------------------------------
        # 6) Extract the control region where the anomaly will be fused.
        #    This local region is used for intensity normalization.
        # ------------------------------------------------------------
        bg_slice = ctrl[(slice(None), *insert_slices)]
        crop_shape = bg_slice.shape[1:]
        crop_to_bg = tuple(slice(0, int(size)) for size in crop_shape)

        if self.params.fusion_keep_bg:
            control_bg_mask = control_background_mask(
                ctrl,
                self.params.fusion_bg_value,
                self.params.fusion_relative_bg_threshold,
                self.params.fusion_bg_exterior_only,
            )
            bg_mask = control_bg_mask[insert_slices]
            target_mask = target_mask.copy()
            target_mask[crop_to_bg] = np.where(bg_mask, 0, target_mask[crop_to_bg])

        # ------------------------------------------------------------
        # 7) Create a spatial valid mask for anomaly foreground.
        # ------------------------------------------------------------
        # Use max projection over channels to define the anomaly texture
        # that feeds edge-aware alpha mask creation.
        anom_proj = np.max(anom, axis=0)
        valid_mask = target_mask > 0

        # ------------------------------------------------------------
        # 8) Create alpha mask from the target mask (or optionally
        #    from Sobel edges) plus distance transform
        #    before normalization so relation restoration
        #    can compensate for the later alpha blending.
        # ------------------------------------------------------------
        alpha_mask = alpha_builder(anom_proj, self.params, target_mask)

        # ------------------------------------------------------------
        # 9) Locally normalize anomaly values while optionally preserving the extracted
        #    anomaly/context relation in the final alpha-blended result.
        # ------------------------------------------------------------
        anom = self._match_local_intensity(
            anom,
            ctrl,
            bg_slice,
            valid_mask,
            target_mask,
            anomaly_roi,
            anomaly_roi_mask,
            alpha_mask,
            self.params,
            normalization_eps=extraction_config.normalization_eps,
        )
        alpha = alpha_mask[None, ...]

        # ------------------------------------------------------------
        # 10) Crop anomaly and alpha to exactly match the clamped
        #     insertion region. This handles anomalies near image borders.
        # ------------------------------------------------------------
        output_slices = tuple(slice(int(offset[axis]), int(offset[axis] + crop_shape[axis])) for axis in range(spatial_ndim))
        anom_crop = anom[(slice(None), *crop_to_bg)]
        alpha_crop = alpha[(slice(None), *crop_to_bg)]

        # ------------------------------------------------------------
        # 11) Alpha blending:
        #     fused = anomaly * alpha + background * (1 - alpha)
        # ------------------------------------------------------------
        fused_region = anom_crop * alpha_crop + bg_slice * (1.0 - alpha_crop)

        # Write fused region back into a copy of the control image.
        fused_image = ctrl.copy()
        fused_image[(slice(None), *output_slices)] = fused_region

        # ------------------------------------------------------------
        # 12) Create segmentation mask in control coordinates.
        # ------------------------------------------------------------
        segmentation = np.zeros(tuple(ctrl_spatial), dtype=np.uint8)
        segmentation[output_slices] = target_mask[crop_to_bg].astype(np.uint8, copy=False)
        segmentation = segmentation[None, ...]
        if channels != 1:
            segmentation = np.repeat(segmentation, channels, axis=0)

        if self.params.fusion_keep_bg:
            fused_image, segmentation = keep_control_background_after_fusion(
                fused_image,
                segmentation,
                ctrl,
                control_bg_mask,
            )

        # Empty target masks intentionally return the unchanged control
        # sample and no ROI debug crops.
        if np.sum(segmentation) == 0:
            return FusionOutput(image=ctrl, segmentation=segmentation)

        # ------------------------------------------------------------
        # 13) Extract ROI around the inserted anomaly for visual checks
        #     and ROI-level evaluation.
        # ------------------------------------------------------------
        centroid = tuple(float(offset[axis]) + float(crop_shape[axis]) / 2.0 for axis in range(spatial_ndim))
        roi_config = extraction_config.roi
        if roi_config.fixed_size is None:
            roi_size = dynamic_roi_size(
                crop_shape,
                roi_config.min_padding,
                roi_config.padding_ratio,
                roi_config.min_size,
            )
        else:
            roi_size = roi_config.fixed_size

        fused_roi = crop_roi(
            fused_image,
            centroid,
            roi_size,
            centroid_is_normalized=False,
        )
        fused_roi_mask = crop_roi(
            segmentation,
            centroid,
            roi_size,
            centroid_is_normalized=False,
        )

        return FusionOutput(
            image=fused_image,
            segmentation=segmentation,
            roi=fused_roi,
            roi_mask=fused_roi_mask,
        )

    @staticmethod
    def _match_local_intensity(
        anom,
        ctrl,
        bg_slice,
        valid_mask,
        target_mask,
        anomaly_roi,
        anomaly_roi_mask,
        alpha_mask,
        params,
        normalization_eps=1e-8,
    ):
        """Match anomaly intensity to local context, optionally restoring the original ROI relation."""
        binary_mask = valid_mask > 0
        normalization_border_width = getattr(params, "fusion_normalization_border_width", 2)
        if normalization_border_width is None or not np.any(binary_mask):
            return anom

        border_width = int(normalization_border_width)
        original_relations = None
        eps = float(normalization_eps)
        output_intensity_bounds = _infer_output_intensity_bounds(ctrl)
        min_context_size = int(getattr(params, "fusion_relation_min_context_size", 8))
        relation_mode = getattr(params, "fusion_relation_mode", "delta")
        norm_classes_separately = bool(getattr(params, "fusion_relation_norm_classes_separately", False))

        if border_width == -1:
            context_slice = ctrl
            context_mask = np.ones(ctrl.shape[1:], dtype=bool)
            fallback_context_mask = context_mask
            dilation_structure = None
        elif border_width >= 0:
            dilation_kernel_size = border_width * 2 + 1
            dilation_structure = np.ones((dilation_kernel_size,) * binary_mask.ndim, dtype=bool)
            context_slice = bg_slice
            fallback_context_mask = ~binary_mask
            dilated_mask = binary_dilation(binary_mask, structure=dilation_structure)
            context_mask = dilated_mask & fallback_context_mask
            if np.count_nonzero(context_mask) < min_context_size:
                context_mask = fallback_context_mask
            if getattr(params, "fusion_restore_anomaly_bg_relation", None):
                original_relations = _anomaly_context_relations(
                    anomaly_roi,
                    anomaly_roi_mask,
                    border_width,
                    relation_mode,
                    eps=eps,
                    min_context_size=min_context_size,
                    norm_classes_separately=norm_classes_separately,
                )
        else:
            raise ValueError("fusion_normalization_border_width must be None, -1, or >= 0.")

        labels = np.unique(target_mask[binary_mask])
        labels = labels[labels > 0]

        if not norm_classes_separately:
            if np.count_nonzero(context_mask) < min_context_size:
                return anom
            return _normalize_anomaly_to_context(
                anom,
                context_slice,
                bg_slice,
                binary_mask,
                context_mask,
                original_relations,
                relation_mode,
                class_label=None,
                alpha_mask=alpha_mask,
                eps=eps,
                output_intensity_bounds=output_intensity_bounds,
            )

        matched = anom
        for label_value in labels:
            class_mask = target_mask == label_value
            if not np.any(class_mask):
                continue
            if border_width == -1:
                class_context_mask = context_mask
            else:
                class_dilated_mask = binary_dilation(class_mask, structure=dilation_structure)
                class_context_mask = class_dilated_mask & ~binary_mask
                if np.count_nonzero(class_context_mask) < min_context_size:
                    class_context_mask = context_mask
                if np.count_nonzero(class_context_mask) < min_context_size:
                    class_context_mask = fallback_context_mask
            if np.count_nonzero(class_context_mask) < min_context_size:
                continue
            matched = _normalize_anomaly_to_context(
                matched,
                context_slice,
                bg_slice,
                class_mask,
                class_context_mask,
                original_relations,
                relation_mode,
                class_label=label_value,
                alpha_mask=alpha_mask,
                eps=eps,
                output_intensity_bounds=output_intensity_bounds,
            )

        return matched
