from __future__ import annotations

from dataclasses import asdict, replace

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from hybrid_sample_generator.fusion.classical.backend import ClassicalFusionBackend
from hybrid_sample_generator.fusion.preprocessing import (
    denormalize_anomaly as _denormalize_anomaly,
    inverse_extraction_scale as _inverse_extraction_scale,
    validate_position as _validate_position,
)
from hybrid_sample_generator.fusion.interfaces import FusionOutput, control_background_mask, keep_control_background_after_fusion
from hybrid_sample_generator.fusion.learned_residual_alpha.configuration import Config
from hybrid_sample_generator.fusion.learned_residual_alpha.model import ResidualAlphaRefiner
from hybrid_sample_generator.fusion.learned_residual_alpha.preprocessing import (
    bbox_slices as _bbox_slices,
    channel_min as _channel_min,
    pseudo_inpaint as _pseudo_inpaint,
    safe_scale as _safe_scale,
    soft_alpha as _soft_alpha,
    spatial_label_mask as _spatial_label_mask,
    support_mask as _support_mask,
    to_tensor as _to_tensor,
    unpack_sample as _unpack_sample,
)
from hybrid_sample_generator.fusion.learned_residual_alpha.training import train_backend
from hybrid_sample_generator.imaging.roi import (
    crop_cube_clip,
    crop_square_clip,
    dynamic_roi_size,
)


class LearnedResidualAlphaFusionBackend:
    """
    Trainable fusion backend that refines a deterministic alpha-blend proposal.

    The model predicts:
      - a bounded correction to the base alpha mask
      - a bounded residual image correction inside a dilated support region

    Training uses real anomalous samples in a self-supervised way: the anomaly
    mask is blurred/inpainted to create a pseudo-background, and the original
    image is used as the reconstruction target.
    """

    def __init__(self, fusion_params: Config | None = None, **kwargs) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown LearnedResidualAlphaFusionBackend parameters: {unknown}")
        if fusion_params is not None and not isinstance(fusion_params, Config):
            raise TypeError(f"fusion_params must be {Config.__module__}.Config.")
        self.params = Config() if fusion_params is None else replace(fusion_params)
        self.params.validate()
        self._parameters_provided = fusion_params is not None
        self.model: ResidualAlphaRefiner | None = None
        self.image_channels: int | None = None
        self.spatial_dims: int | None = None
        self.device = torch.device("cpu")

    def warmup(self, shape, device=None, dtype=None, config=None):
        self.params.validate()
        if len(shape) not in (3, 4):
            raise ValueError(f"Expected channel-first 2D/3D shape, got {shape!r}.")
        spatial_dims = len(shape) - 1
        configured_dims = self.params.spatial_dims
        if configured_dims is not None and int(configured_dims) != spatial_dims:
            raise ValueError(
                f"LearnedResidualAlphaFusionBackend configured for {configured_dims}D, "
                f"but got sample shape {shape!r}."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        image_channels = int(shape[0])
        input_channels = 3 * image_channels + 3
        if (
            self.model is None
            or self.image_channels != image_channels
            or self.spatial_dims != spatial_dims
        ):
            self.model = ResidualAlphaRefiner(
                input_channels=input_channels,
                image_channels=image_channels,
                spatial_dims=spatial_dims,
                base_channels=int(self.params.base_channels),
                depth=int(self.params.depth),
            )
            self.image_channels = image_channels
            self.spatial_dims = spatial_dims

        self.model.to(self.device)
        if dtype is not None:
            self.model.to(dtype=dtype)
        return self

    def save_checkpoint(self, path: str, **extra_state) -> None:
        if self.model is None:
            raise ValueError("Cannot save fusion backend before warmup/training.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "params": asdict(self.params),
                "image_channels": self.image_channels,
                "spatial_dims": self.spatial_dims,
                **extra_state,
            },
            path,
        )

    def load_checkpoint(self, path: str, **kwargs) -> None:
        state = torch.load(path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        params = state.get("params")
        if params is not None:
            saved_params = Config(**params)
            saved_params.validate()
            if not self._parameters_provided:
                self.params = saved_params
            else:
                for name in ("base_channels", "depth"):
                    if getattr(self.params, name) != getattr(saved_params, name):
                        raise ValueError(f"Fusion checkpoint architecture conflicts with configured {name}.")

        image_channels = int(state.get("image_channels", self.image_channels or 1))
        spatial_dims = int(state.get("spatial_dims", self.params.spatial_dims or 2))
        self.model = None
        self.warmup((image_channels, *((1,) * spatial_dims)), device=kwargs.get("device"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

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
        return train_backend(
            self,
            sample_dataloader,
            epochs=epochs,
            lr=lr,
            checkpoint_path=checkpoint_path,
            device=device,
            config=config,
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
                "LearnedResidualAlphaFusionBackend requires extraction_config for ROI construction."
            )
        control = control_img
        anomaly = sample["synth_anomaly"]
        anomaly_meta = sample["anomaly_meta"]
        target_mask = sample["tgt_mask"]
        control_bg_mask = None
        if self.params.fusion_keep_bg:
            control_bg_mask = control_background_mask(
                control,
                self.params.fusion_bg_value,
                self.params.fusion_relative_bg_threshold,
                self.params.fusion_bg_exterior_only,
            )
        proposal = self._prepare_fusion_proposal(
            control,
            anomaly,
            anomaly_meta,
            position,
            target_mask,
            control_bg_mask=control_bg_mask,
            normalization_eps=extraction_config.normalization_eps,
        )
        spatial_dims = proposal["spatial_dims"]

        self.warmup(proposal["control"].shape, config=extraction_config)
        self.model.eval()

        features, scale = self._build_features(
            proposal["bg_slice"],
            proposal["anomaly_crop"],
            proposal["base_alpha"],
            proposal["support_mask"],
        )
        with torch.no_grad():
            fused_region, alpha_delta, residual = self._forward_components(
                features,
                _to_tensor(proposal["bg_slice"], self.device),
                _to_tensor(proposal["anomaly_crop"], self.device),
                _to_tensor(proposal["base_alpha"][None, ...], self.device),
                _to_tensor(proposal["support_mask"][None, ...], self.device),
                torch.as_tensor(scale, dtype=torch.float32, device=self.device).view(1, 1, *([1] * spatial_dims)),
            )

        fused_region_np = fused_region.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        fused_image = proposal["control"].copy()
        fused_image[(slice(None), *proposal["output_slices"])] = fused_region_np
        if self.params.clamp_output:
            fused_image = np.clip(fused_image, 0.0, 1.0)

        segmentation = np.zeros(tuple(proposal["control"].shape[1:]), dtype=np.uint8)
        segmentation[proposal["output_slices"]] = proposal["mask_crop"].astype(np.uint8, copy=False)
        segmentation = segmentation[None, ...]
        if proposal["control"].shape[0] != 1:
            segmentation = np.repeat(segmentation, proposal["control"].shape[0], axis=0)

        if self.params.fusion_keep_bg:
            fused_image, segmentation = keep_control_background_after_fusion(
                fused_image,
                segmentation,
                proposal["control"],
                proposal["control_bg_mask"],
            )

        if np.sum(segmentation) == 0:
            return FusionOutput(image=proposal["control"], segmentation=segmentation)

        crop_shape = proposal["bg_slice"].shape[1:]
        centroid = tuple(
            float(proposal["offset"][axis]) + float(crop_shape[axis]) / 2.0
            for axis in range(spatial_dims)
        )
        roi_config = extraction_config.roi
        if roi_config.fixed_size is None:
            if spatial_dims == 2:
                roi_size = dynamic_roi_size(
                    crop_shape,
                    roi_config.min_padding,
                    roi_config.padding_ratio,
                    roi_config.min_size,
                )
                crop_roi = crop_square_clip
            else:
                roi_size = dynamic_roi_size(
                    crop_shape,
                    roi_config.min_padding,
                    roi_config.padding_ratio,
                    roi_config.min_size,
                )
                crop_roi = crop_cube_clip
        else:
            roi_size = roi_config.fixed_size
            crop_roi = crop_square_clip if spatial_dims == 2 else crop_cube_clip

        return FusionOutput(
            image=fused_image,
            segmentation=segmentation,
            roi=crop_roi(fused_image, centroid, roi_size, centroid_is_normalized=False),
            roi_mask=crop_roi(segmentation, centroid, roi_size, centroid_is_normalized=False),
            metrics={
                "alpha_delta_abs_mean": float(torch.mean(torch.abs(alpha_delta)).detach().cpu().item()),
                "residual_abs_mean": float(torch.mean(torch.abs(residual)).detach().cpu().item()),
            },
        )

    def _forward_components(self, features, control, anomaly, base_alpha, support, scale):
        alpha_delta, residual = self.model(features)
        alpha_delta = torch.tanh(alpha_delta) * float(self.params.alpha_delta_scale)
        residual = torch.tanh(residual) * float(self.params.residual_scale) * scale
        final_alpha = torch.clamp(base_alpha + alpha_delta * support, 0.0, 1.0)
        fused = final_alpha * anomaly + (1.0 - final_alpha) * control + residual * support
        return fused, alpha_delta, residual

    def _training_loss(self, fused, target, alpha_delta, residual, mask, support):
        per_pixel = F.smooth_l1_loss(fused, target, reduction="none")
        weights = (
            1.0
            + mask * float(self.params.foreground_loss_weight)
            + support * float(self.params.support_loss_weight)
        )
        recon_loss = torch.mean(per_pixel * weights)
        alpha_reg = torch.mean(torch.abs(alpha_delta)) * float(self.params.alpha_delta_l1)
        residual_reg = torch.mean(torch.abs(residual)) * float(self.params.residual_l1)
        return recon_loss + alpha_reg + residual_reg

    def _prepare_training_sample(self, sample):
        img, seg, _basename = _unpack_sample(sample)
        img = np.asarray(img, dtype=np.float32)
        seg = np.asarray(seg)
        if img.ndim not in (3, 4):
            raise ValueError(f"Expected channel-first 2D/3D sample, got {img.shape!r}.")

        spatial_dims = img.ndim - 1
        configured_dims = self.params.spatial_dims
        if configured_dims is not None and int(configured_dims) != spatial_dims:
            return None

        mask = _spatial_label_mask(seg, spatial_dims) > 0
        if not np.any(mask):
            return None

        crop_slices = _bbox_slices(mask, margin=int(self.params.train_crop_margin), shape=mask.shape)
        target = img[(slice(None), *crop_slices)].astype(np.float32, copy=False)
        mask_crop = mask[crop_slices].astype(np.float32, copy=False)
        control = _pseudo_inpaint(target, mask_crop, sigma=float(self.params.train_inpaint_blur_sigma))
        anomaly = np.where(mask_crop[None, ...] > 0, target, _channel_min(target))
        base_alpha = _soft_alpha(mask_crop, self.params, spatial_dims)
        support = _support_mask(mask_crop, self.params.residual_border_width, spatial_dims)

        features, scale = self._build_features(control, anomaly, base_alpha, support)
        return (
            features,
            _to_tensor(target, self.device),
            _to_tensor(control, self.device),
            _to_tensor(anomaly, self.device),
            _to_tensor(base_alpha[None, ...], self.device),
            _to_tensor(support[None, ...], self.device),
            _to_tensor(mask_crop[None, ...], self.device),
            torch.as_tensor(scale, dtype=torch.float32, device=self.device).view(1, 1, *([1] * spatial_dims)),
        )

    def _prepare_fusion_proposal(
        self,
        control,
        anomaly,
        anomaly_meta,
        position,
        target_mask,
        *,
        control_bg_mask=None,
        normalization_eps=1e-8,
    ):
        if anomaly_meta is None:
            raise ValueError("anomaly_meta must be provided (needs at least 'scale_factor').")
        scale_factor = anomaly_meta.get("scale_factor")
        if scale_factor is None:
            raise ValueError("anomaly_meta is missing required key 'scale_factor'.")
        if target_mask is None:
            raise ValueError("LearnedResidualAlphaFusionBackend requires target_mask.")

        ctrl = np.asarray(control, dtype=np.float32)
        anom = _denormalize_anomaly(np.asarray(anomaly, dtype=np.float32), anomaly_meta)
        spatial_dims = ctrl.ndim - 1
        if spatial_dims not in (2, 3):
            raise ValueError(f"Expected channel-first 2D/3D control sample, got {ctrl.shape!r}.")
        if anom.ndim != ctrl.ndim:
            raise ValueError(f"control and anomaly must have same ndim. Got {ctrl.shape} and {anom.shape}.")

        target_mask = _spatial_label_mask(np.asarray(target_mask), spatial_dims).astype(np.uint8, copy=False)
        if target_mask.shape != anom.shape[1:]:
            raise ValueError(f"target_mask shape {target_mask.shape} does not match anomaly shape {anom.shape[1:]}.")

        foreground = target_mask > 0
        if np.any(foreground):
            crop_slices = _bbox_slices(foreground, margin=0, shape=foreground.shape)
            anom = anom[(slice(None), *crop_slices)]
            target_mask = target_mask[crop_slices]

        scale = _inverse_extraction_scale(scale_factor, ndim=spatial_dims)
        anom = zoom(anom, (1.0, *scale), order=1)
        target_mask = zoom(target_mask, scale, order=0).astype(np.uint8, copy=False)

        position = _validate_position(position, spatial_dims)
        ctrl_spatial = np.array(ctrl.shape[1:], dtype=int)
        anom_spatial = np.array(anom.shape[1:], dtype=int)
        offset = np.array(
            [round(ctrl_spatial[axis] * position[axis] - anom_spatial[axis] / 2) for axis in range(spatial_dims)],
            dtype=int,
        )
        offset_end = offset + anom_spatial
        for axis, (start, end, limit) in enumerate(zip(offset, offset_end, ctrl_spatial)):
            if end > limit:
                shift = end - limit
                offset[axis] -= shift
                offset_end[axis] -= shift
            if offset[axis] < 0:
                shift = -offset[axis]
                offset[axis] += shift
                offset_end[axis] += shift

        output_slices = tuple(slice(int(start), int(end)) for start, end in zip(offset, offset_end))
        bg_slice = ctrl[(slice(None), *output_slices)]
        crop_shape = bg_slice.shape[1:]
        crop_to_bg = tuple(slice(0, int(size)) for size in crop_shape)

        if self.params.fusion_keep_bg:
            if control_bg_mask is None:
                raise ValueError("control_bg_mask is required when fusion_keep_bg=True.")
            bg_mask = control_bg_mask[output_slices]
            target_mask = target_mask.copy()
            target_mask[crop_to_bg] = np.where(bg_mask, 0, target_mask[crop_to_bg])

        mask_crop = target_mask[crop_to_bg].astype(np.float32, copy=False)
        base_alpha = _soft_alpha(mask_crop, self.params, spatial_dims)
        alpha_for_normalization = np.zeros_like(target_mask, dtype=np.float32)
        alpha_for_normalization[crop_to_bg] = base_alpha
        anom = ClassicalFusionBackend._match_local_intensity(
            anom,
            ctrl,
            bg_slice,
            target_mask > 0,
            target_mask,
            None,
            None,
            alpha_for_normalization,
            self.params,
            normalization_eps=normalization_eps,
        )

        anomaly_crop = anom[(slice(None), *crop_to_bg)]
        support_mask = _support_mask(mask_crop, self.params.residual_border_width, spatial_dims)

        return {
            "control": ctrl,
            "bg_slice": bg_slice,
            "anomaly_crop": anomaly_crop,
            "mask_crop": mask_crop,
            "base_alpha": base_alpha,
            "support_mask": support_mask,
            "output_slices": output_slices,
            "offset": offset,
            "spatial_dims": spatial_dims,
            "control_bg_mask": control_bg_mask,
        }

    def _build_features(self, control, anomaly, base_alpha, support):
        base_fused = base_alpha[None, ...] * anomaly + (1.0 - base_alpha[None, ...]) * control
        scale = _safe_scale(control)
        center = np.mean(control, dtype=np.float32)
        image_features = np.concatenate(
            [
                (control - center) / scale,
                (anomaly - center) / scale,
                (base_fused - center) / scale,
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        feature_np = np.concatenate(
            [
                image_features,
                base_alpha[None, ...].astype(np.float32, copy=False),
                (base_alpha > 1e-4)[None, ...].astype(np.float32, copy=False),
                support[None, ...].astype(np.float32, copy=False),
            ],
            axis=0,
        )
        return _to_tensor(feature_np, self.device), np.float32(scale)
