from __future__ import annotations

import numpy as np
import scipy.ndimage
from scipy.ndimage import binary_dilation, zoom

from fusion_backend.classical.configuration import CONFIDENCE_LEVELS, Config
from fusion_backend.fusion_configuration import FusionConfiguration
from fusion_backend.interfaces import FusionOutput
from synthesizer.functions_2D.Anomaly_Extraction2D import crop_square_clip, dynamic_roi_size as dynamic_roi_size_2d
from synthesizer.functions_3D.Anomaly_Extraction3D import crop_cube_clip, dynamic_roi_size as dynamic_roi_size_3d


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

    def __init__(self, fusion_params=None, **kwargs) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown ClassicalFusionBackend parameters: {unknown}")
        self.params = _normalize_params(fusion_params)

    def warmup(self, shape, device=None, dtype=None, config=None):
        return self

    def load_checkpoint(self, path: str, **kwargs) -> None:
        return None

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
        print("ClassicalFusionBackend is not trainable. Skipping fusion backend training.")
        return {
            "skipped": True,
            "reason": "ClassicalFusionBackend is not trainable.",
            "checkpoint_path": None,
        }

    def fuse(
        self,
        control,
        anomaly,
        anomaly_meta,
        position,
        *,
        target_mask,
        config=None,
        **kwargs,
    ) -> FusionOutput:
        if config is None:
            raise ValueError("ClassicalFusionBackend requires config for fusion parameters.")

        if control.ndim == 3:
            return self._fuse_spatial(
                control,
                anomaly,
                anomaly_meta,
                position,
                target_mask,
                config,
                spatial_ndim=2,
                crop_roi=crop_square_clip,
                dynamic_roi_size=dynamic_roi_size_2d,
                alpha_builder=_get_alpha_mask_2d,
            )
        if control.ndim == 4:
            return self._fuse_spatial(
                control,
                anomaly,
                anomaly_meta,
                position,
                target_mask,
                config,
                spatial_ndim=3,
                crop_roi=crop_cube_clip,
                dynamic_roi_size=dynamic_roi_size_3d,
                alpha_builder=_get_alpha_mask_3d,
            )
        raise ValueError(f"Unexpected shape: {control.shape}, Supported: (C, H, W) or (C, D, H, W)")

    def _fuse_spatial(
        self,
        control,
        anomaly,
        anomaly_meta,
        position,
        target_mask,
        config,
        *,
        spatial_ndim: int,
        crop_roi,
        dynamic_roi_size,
        alpha_builder,
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
        anom_min = float(np.nanmin(anom))
        anom = np.where(target_mask > 0, anom, anom_min)

        # ------------------------------------------------------------
        # 2) Trim spatial padding by cropping to the foreground bounding
        #    box. The same crop is used for all channels.
        # ------------------------------------------------------------
        # Foreground mask over spatial dims: target mask defines the
        # intended label footprint.
        foreground = _spatial_label_mask(target_mask, spatial_ndim) > 0
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
        anom = zoom(anom, (1.0, *scale), order=1)
        if target_mask.ndim == expected_ndim:
            target_mask = zoom(target_mask, (1.0, *scale), order=0)
        else:
            target_mask = zoom(target_mask, scale, order=0)
        target_mask = _spatial_label_mask(target_mask, spatial_ndim).astype(np.uint8, copy=False)

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

        # ------------------------------------------------------------
        # 7) Create a spatial valid mask for anomaly foreground.
        # ------------------------------------------------------------
        # Use max projection over channels to define the anomaly texture
        # that feeds edge-aware alpha mask creation.
        anom_proj = np.max(anom, axis=0)
        valid_mask = target_mask > 0

        # ------------------------------------------------------------
        # 8) Locally normalize anomaly values per channel to match the
        #    target area plus optional surrounding border.
        # ------------------------------------------------------------
        anom = self._match_local_intensity(anom, ctrl, bg_slice, valid_mask, self.params)

        # ------------------------------------------------------------
        # 9) Create alpha mask from the target mask (or optionally
        #    from Sobel edges) plus distance transform.
        # ------------------------------------------------------------
        alpha_mask = alpha_builder(anom_proj, self.params, target_mask)
        alpha = alpha_mask[None, ...]

        # ------------------------------------------------------------
        # 10) Crop anomaly and alpha to exactly match the clamped
        #     insertion region. This handles anomalies near image borders.
        # ------------------------------------------------------------
        crop_shape = bg_slice.shape[1:]
        output_slices = tuple(slice(int(offset[axis]), int(offset[axis] + crop_shape[axis])) for axis in range(spatial_ndim))
        crop_to_bg = tuple(slice(0, int(size)) for size in crop_shape)
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

        # Empty target masks intentionally return the unchanged control
        # sample and no ROI debug crops.
        if np.sum(segmentation) == 0:
            return FusionOutput(image=ctrl, segmentation=segmentation)

        # ------------------------------------------------------------
        # 13) Extract ROI around the inserted anomaly for visual checks
        #     and ROI-level evaluation.
        # ------------------------------------------------------------
        centroid = tuple(float(offset[axis]) + float(crop_shape[axis]) / 2.0 for axis in range(spatial_ndim))
        if config.fixed_roi_size is None:
            roi_size = dynamic_roi_size(crop_shape, config.min_pad, config.pad_ratio, config.min_roi_size)
        else:
            roi_size = config.fixed_roi_size

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
    def _match_local_intensity(anom, ctrl, bg_slice, valid_mask, params):
        """Match anomaly intensity statistics to the target control region."""
        binary_mask = valid_mask > 0
        normalization_border_width = params["fusion_normalization_border_width"]
        if normalization_border_width is None or not np.any(binary_mask):
            return anom

        border_width = int(normalization_border_width)
        if border_width == -1:
            normalization_mask = None
        elif border_width >= 0:
            dilation_kernel_size = border_width * 2 + 1
            dilation_structure = np.ones((dilation_kernel_size,) * binary_mask.ndim, dtype=bool)
            normalization_mask = binary_dilation(binary_mask, structure=dilation_structure)
        else:
            raise ValueError("fusion_normalization_border_width must be None, -1, or >= 0.")

        if border_width != -1 and not np.any(normalization_mask):
            return anom

        matched = anom.copy()
        for channel in range(matched.shape[0]):
            # Anomaly intensity inside the intended foreground.
            anomaly_values = matched[channel][binary_mask]
            if anomaly_values.size == 0:
                continue

            # Background intensity either from the whole control image
            # or from the local region around the target mask.
            background_values = ctrl[channel].ravel() if border_width == -1 else bg_slice[channel][normalization_mask]
            if background_values.size == 0:
                continue

            # Mean/std matching in z-score space.
            anomaly_std = np.std(anomaly_values)
            if anomaly_std == 0:
                anomaly_std = 1e-5

            matched[channel][binary_mask] = (
                (anomaly_values - np.mean(anomaly_values)) / anomaly_std
            ) * np.std(background_values) + np.mean(background_values)

        return matched


def _inverse_extraction_scale(scale_factor, ndim):
    """Return the spatial zoom that restores an extraction resize."""
    if np.isscalar(scale_factor):
        scale = np.full(ndim, float(scale_factor), dtype=np.float32)
    else:
        scale = np.asarray(scale_factor, dtype=np.float32).reshape(-1)
        if scale.size != ndim:
            raise ValueError(f"scale_factor must have len {ndim}. Got {scale_factor!r}")

    if np.any(scale <= 0):
        raise ValueError(f"scale_factor values must be > 0. Got {scale_factor!r}")

    return tuple(float(1.0 / value) for value in scale)


def _denormalize_anomaly(anomaly, normalization_meta):
    if not normalization_meta:
        return anomaly

    norm_type = normalization_meta.get("norm_type")
    if norm_type == "zscore":
        mean = normalization_meta.get("norm_mean")
        std = normalization_meta.get("norm_std")
        if mean is None or std is None:
            return anomaly
        return anomaly * float(std) + float(mean)
    if norm_type == "zscore_median":
        median = normalization_meta.get("norm_median")
        mad = normalization_meta.get("norm_mad")
        if median is None or mad is None:
            return anomaly
        return anomaly * float(mad) + float(median)

    return anomaly


def _spatial_label_mask(mask, spatial_ndim):
    mask = np.asarray(mask)
    if mask.ndim == spatial_ndim:
        return mask
    if mask.ndim == spatial_ndim + 1:
        return np.max(mask, axis=0)
    raise ValueError(f"target mask must have {spatial_ndim} or {spatial_ndim + 1} dims. Got {mask.shape}")


def _validate_position(position, spatial_ndim):
    if position is None:
        axes = "(H,W)" if spatial_ndim == 2 else "(D,H,W)"
        raise ValueError(f"position must be provided (expected len {spatial_ndim}: {axes}).")
    position = list(position)
    if len(position) != spatial_ndim:
        axes = "(H,W)" if spatial_ndim == 2 else "(D,H,W)"
        raise ValueError(f"position must have len {spatial_ndim} {axes}. Got {position!r}")
    return tuple(float(value) for value in position)


def _sample_alpha_params(config):
    max_alpha = config["max_alpha"]
    sq = config["sq"]
    steepness_factor = config["steepness_factor"]

    if config["fusion_variation"]:
        confidence_z_score = _confidence_z_score(config)

        std_alpha = config["alpha_variation"] / confidence_z_score
        max_alpha = float(np.clip(np.random.normal(max_alpha, std_alpha), 0.0, 1.0))

        std_sq = config["sq_variation"] / confidence_z_score
        sq = float(np.maximum(0.1, np.random.normal(sq, std_sq)))

        std_steepness = config["steepness_variation"] / confidence_z_score
        steepness_factor = float(np.maximum(0.1, np.random.normal(steepness_factor, std_steepness)))

    return max_alpha, sq, steepness_factor


def _get_alpha_mask_2d(anomaly_arr, config, valid_mask):
    """
    Build an alpha blending mask for a 2D anomaly image using:
      - target mask or alternatively sobel edges to detect boundaries
      - if sobel was used: morphological operations to close gaps and remove noise
      - distance transform to produce a smooth mask interior
      - non-linear shaping to control blending strength and falloff
    """
    max_alpha, sq, steepness_factor = _sample_alpha_params(config)

    # Initialize alpha mask.
    alpha_mask = np.zeros_like(anomaly_arr, dtype=np.float32)

    # Skip if no foreground content exists.
    if not np.any(valid_mask > 0):
        return alpha_mask

    if config.get("fusion_use_sobel_for_alpha_mask", False):
        final_clean_mask = _clean_edge_mask(anomaly_arr, valid_mask, config)
    else:
        final_clean_mask = valid_mask > 0
    if not np.any(final_clean_mask):
        return alpha_mask

    alpha_mask[:, :] = _distance_alpha(final_clean_mask, max_alpha, sq, steepness_factor, config)
    return alpha_mask


def _get_alpha_mask_3d(anomaly_arr, config, valid_mask):
    """
    Build a per-slice alpha blending mask for a 3D anomaly volume.

    The mask is computed slice-by-slice along the first spatial axis
    (assumed to be D), using the same edge-aware distance-transform logic
    as the 2D path.
    """
    max_alpha, sq, steepness_factor = _sample_alpha_params(config)

    # Initialize alpha mask volume.
    alpha_mask = np.zeros_like(anomaly_arr, dtype=np.float32)

    # Iterate through slices along D.
    for depth in range(anomaly_arr.shape[0]):
        valid_slice = valid_mask[depth, :, :]

        # Skip slices with no foreground content.
        if not np.any(valid_slice > 0):
            continue

        if config.get("fusion_use_sobel_for_alpha_mask", False):
            final_clean_mask = _clean_edge_mask(anomaly_arr[depth, :, :], valid_slice, config)
        else:
            final_clean_mask = valid_slice > 0
        if not np.any(final_clean_mask):
            continue

        alpha_mask[depth, :, :] = _distance_alpha(final_clean_mask, max_alpha, sq, steepness_factor, config)

    return alpha_mask


def _clean_edge_mask(slice_img, valid_mask, config):
    """
    Create a clean binary support mask from image edges and the target mask.

    This reproduces the old Sobel + morphology pipeline:
      1) detect edges
      2) close gaps and fill interior
      3) optionally shave boundary pixels to reduce blending artifacts
      4) enforce the explicit target mask as the final support
    """
    # ------------------------------------------------------------
    # Build structuring elements for morphology operations.
    # ------------------------------------------------------------
    dilation_size = config["dilation_size"]

    # Circular 2D structuring element ("round brush") used to close gaps.
    y, x = np.ogrid[-dilation_size: dilation_size + 1, -dilation_size: dilation_size + 1]
    struct_brush = x ** 2 + y ** 2 <= dilation_size ** 2

    # 3x3-ish structure used for "shaving" boundary pixels.
    struct_shave = scipy.ndimage.generate_binary_structure(2, 2)

    # ------------------------------------------------------------
    # 1) Sobel edge detection.
    # ------------------------------------------------------------
    grad_y = scipy.ndimage.sobel(slice_img, axis=0)
    grad_x = scipy.ndimage.sobel(slice_img, axis=1)
    grad_mag = np.hypot(grad_y, grad_x)

    # Normalize gradient magnitude to [0, 1].
    if grad_mag.max() > 0:
        grad_mag /= grad_mag.max()

    # Edge mask = pixels with gradient magnitude above threshold.
    edge_mask = grad_mag > config["sobel_threshold"]

    # ------------------------------------------------------------
    # 2) Close gaps and fill the interior.
    # ------------------------------------------------------------
    thick_edge = scipy.ndimage.binary_dilation(edge_mask, structure=struct_brush)
    filled_body = scipy.ndimage.binary_fill_holes(thick_edge)

    # Erode once to compensate for dilation thickening.
    restored_mask = scipy.ndimage.binary_erosion(filled_body, structure=struct_brush, iterations=1)

    # ------------------------------------------------------------
    # 3) Optional "shaving" to remove boundary pixels.
    # ------------------------------------------------------------
    shave_pixels = config["shave_pixels"]
    if shave_pixels > 0:
        clean_mask = scipy.ndimage.binary_erosion(restored_mask, structure=struct_shave, iterations=shave_pixels)
    else:
        clean_mask = restored_mask

    # Enforce the explicit target mask after morphology.
    clean_mask = clean_mask.copy()
    clean_mask[valid_mask <= 0] = False
    return clean_mask


def _distance_alpha(clean_mask, max_alpha, sq, steepness_factor, config):
    """Create smooth interior alpha weights from a clean binary support mask."""
    # ------------------------------------------------------------
    # 4) Distance transform to create smooth interior weights.
    # ------------------------------------------------------------
    upsampling_factor = config["upsampling_factor"]
    if upsampling_factor > 1:
        # Upsample mask for smoother distance transform, then downsample.
        large_mask = scipy.ndimage.zoom(clean_mask, upsampling_factor, order=0)
        large_dist = scipy.ndimage.distance_transform_edt(large_mask)
        dist_map = scipy.ndimage.zoom(large_dist, 1 / upsampling_factor, order=1)
    else:
        dist_map = scipy.ndimage.distance_transform_edt(clean_mask)

    # Normalize to [0, 1].
    current_max = dist_map.max()
    if current_max > 0:
        dist_map /= current_max

    # ------------------------------------------------------------
    # 5) Apply shaping parameters to control falloff and maximum alpha.
    # ------------------------------------------------------------
    dist_map *= steepness_factor
    dist_map = np.clip(dist_map, 0, 1.0)
    dist_map = (dist_map ** sq) * max_alpha

    # Enforce zero outside the clean mask.
    dist_map[~clean_mask] = 0.0
    return dist_map.astype(np.float32, copy=False)


def _normalize_params(fusion_params):
    if fusion_params is None:
        return FusionConfiguration(Config()).fixed_params()
    if isinstance(fusion_params, FusionConfiguration):
        return fusion_params.fixed_params()
    return FusionConfiguration.from_value(fusion_params).fixed_params()


def _confidence_z_score(params):
    selected_confidence = params["selected_confidence"]
    try:
        return CONFIDENCE_LEVELS[selected_confidence]
    except KeyError as exc:
        raise ValueError(
            f"Unknown selected_confidence {selected_confidence!r}. "
            f"Supported values: {list(CONFIDENCE_LEVELS)}"
        ) from exc
