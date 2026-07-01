from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, zoom
from tqdm import tqdm

from fusion_backend.classical.backend import _denormalize_anomaly, _inverse_extraction_scale, _validate_position
from fusion_backend.classical.backend import ClassicalFusionBackend
from fusion_backend.fusion_configuration import FusionConfiguration
from fusion_backend.interfaces import FusionOutput
from fusion_backend.learned_residual_alpha.configuration import Config
from fusion_backend.learned_residual_alpha.model import ResidualAlphaRefiner
from synthesizer.functions_2D.Anomaly_Extraction2D import crop_square_clip, dynamic_roi_size as dynamic_roi_size_2d
from synthesizer.functions_3D.Anomaly_Extraction3D import crop_cube_clip, dynamic_roi_size as dynamic_roi_size_3d


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

    def __init__(self, fusion_params=None, **kwargs) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown LearnedResidualAlphaFusionBackend parameters: {unknown}")
        self.params = _normalize_params(fusion_params)
        self.model: ResidualAlphaRefiner | None = None
        self.image_channels: int | None = None
        self.spatial_dims: int | None = None
        self.device = torch.device("cpu")

    def warmup(self, shape, device=None, dtype=None, config=None):
        if len(shape) not in (3, 4):
            raise ValueError(f"Expected channel-first 2D/3D shape, got {shape!r}.")
        spatial_dims = len(shape) - 1
        configured_dims = self.params.get("spatial_dims")
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
                base_channels=int(self.params["base_channels"]),
                depth=int(self.params["depth"]),
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
                "params": self.params,
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
            self.params.update(params)

        image_channels = int(state.get("image_channels", self.image_channels or 1))
        spatial_dims = int(state.get("spatial_dims", self.params.get("spatial_dims") or 2))
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
        epochs = int(epochs if epochs is not None else self.params["train_epochs"])
        lr = float(lr if lr is not None else self.params["train_lr"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        optimizer = None
        history: list[float] = []
        max_samples = self.params.get("train_max_samples_per_epoch")
        log_every = self.params.get("log_every")

        for epoch in range(epochs):
            losses = []
            iterator = tqdm(sample_dataloader, desc=f"Fusion backend epoch {epoch + 1}/{epochs}")
            for sample_idx, sample in enumerate(iterator):
                if max_samples is not None and sample_idx >= int(max_samples):
                    break

                prepared = self._prepare_training_sample(sample)
                if prepared is None:
                    continue

                features, target, control, anomaly, base_alpha, support, mask, scale = prepared
                if self.model is None:
                    self.warmup(tuple(target.shape[1:]), device=self.device)
                if optimizer is None:
                    optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr=lr,
                        weight_decay=float(self.params["train_weight_decay"]),
                    )
                    self.model.train()

                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                fused, alpha_delta, residual = self._forward_components(
                    features,
                    control,
                    anomaly,
                    base_alpha,
                    support,
                    scale,
                )
                loss = self._training_loss(fused, target, alpha_delta, residual, mask, support)
                loss.backward()
                grad_clip_norm = self.params.get("grad_clip_norm")
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip_norm))
                optimizer.step()

                losses.append(float(loss.detach().cpu().item()))
                if log_every and (sample_idx + 1) % int(log_every) == 0:
                    iterator.set_postfix(loss=f"{np.mean(losses[-int(log_every):]):.5f}")

            if not losses:
                raise ValueError("No valid anomalous samples were found for fusion backend training.")
            mean_loss = float(np.mean(losses))
            history.append(mean_loss)

        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path, train_loss_history=history)

        self.model.eval()
        return {"train_loss_history": history, "checkpoint_path": checkpoint_path}

    def fuse(
        self,
        sample: dict,
        control_img,
        position,
        *,
        config=None,
    ) -> FusionOutput:
        if config is None:
            raise ValueError("LearnedResidualAlphaFusionBackend requires config for ROI parameters.")
        control = control_img
        anomaly = sample["synth_anomaly"]
        anomaly_meta = sample["anomaly_meta"]
        target_mask = sample["tgt_mask"]
        proposal = self._prepare_fusion_proposal(control, anomaly, anomaly_meta, position, target_mask)
        spatial_dims = proposal["spatial_dims"]

        self.warmup(proposal["control"].shape, config=config)
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
        if self.params.get("clamp_output"):
            fused_image = np.clip(fused_image, 0.0, 1.0)

        segmentation = np.zeros(tuple(proposal["control"].shape[1:]), dtype=np.uint8)
        segmentation[proposal["output_slices"]] = proposal["mask_crop"].astype(np.uint8, copy=False)
        segmentation = segmentation[None, ...]
        if proposal["control"].shape[0] != 1:
            segmentation = np.repeat(segmentation, proposal["control"].shape[0], axis=0)

        if np.sum(segmentation) == 0:
            return FusionOutput(image=proposal["control"], segmentation=segmentation)

        crop_shape = proposal["bg_slice"].shape[1:]
        centroid = tuple(
            float(proposal["offset"][axis]) + float(crop_shape[axis]) / 2.0
            for axis in range(spatial_dims)
        )
        if config.fixed_roi_size is None:
            if spatial_dims == 2:
                roi_size = dynamic_roi_size_2d(crop_shape, config.min_pad, config.pad_ratio, config.min_roi_size)
                crop_roi = crop_square_clip
            else:
                roi_size = dynamic_roi_size_3d(crop_shape, config.min_pad, config.pad_ratio, config.min_roi_size)
                crop_roi = crop_cube_clip
        else:
            roi_size = config.fixed_roi_size
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
        alpha_delta = torch.tanh(alpha_delta) * float(self.params["alpha_delta_scale"])
        residual = torch.tanh(residual) * float(self.params["residual_scale"]) * scale
        final_alpha = torch.clamp(base_alpha + alpha_delta * support, 0.0, 1.0)
        fused = final_alpha * anomaly + (1.0 - final_alpha) * control + residual * support
        return fused, alpha_delta, residual

    def _training_loss(self, fused, target, alpha_delta, residual, mask, support):
        per_pixel = F.smooth_l1_loss(fused, target, reduction="none")
        weights = (
            1.0
            + mask * float(self.params["foreground_loss_weight"])
            + support * float(self.params["support_loss_weight"])
        )
        recon_loss = torch.mean(per_pixel * weights)
        alpha_reg = torch.mean(torch.abs(alpha_delta)) * float(self.params["alpha_delta_l1"])
        residual_reg = torch.mean(torch.abs(residual)) * float(self.params["residual_l1"])
        return recon_loss + alpha_reg + residual_reg

    def _prepare_training_sample(self, sample):
        img, seg, _basename = _unpack_sample(sample)
        img = np.asarray(img, dtype=np.float32)
        seg = np.asarray(seg)
        if img.ndim not in (3, 4):
            raise ValueError(f"Expected channel-first 2D/3D sample, got {img.shape!r}.")

        spatial_dims = img.ndim - 1
        configured_dims = self.params.get("spatial_dims")
        if configured_dims is not None and int(configured_dims) != spatial_dims:
            return None

        mask = _spatial_label_mask(seg, spatial_dims) > 0
        if not np.any(mask):
            return None

        crop_slices = _bbox_slices(mask, margin=int(self.params["train_crop_margin"]), shape=mask.shape)
        target = img[(slice(None), *crop_slices)].astype(np.float32, copy=False)
        mask_crop = mask[crop_slices].astype(np.float32, copy=False)
        control = _pseudo_inpaint(target, mask_crop, sigma=float(self.params["train_inpaint_blur_sigma"]))
        anomaly = np.where(mask_crop[None, ...] > 0, target, _channel_min(target))
        base_alpha = _soft_alpha(mask_crop, self.params, spatial_dims)
        support = _support_mask(mask_crop, self.params, spatial_dims)

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

    def _prepare_fusion_proposal(self, control, anomaly, anomaly_meta, position, target_mask):
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
        anom = ClassicalFusionBackend._match_local_intensity(
            anom,
            ctrl,
            bg_slice,
            target_mask > 0,
            target_mask,
            None,
            None,
            None,
            self.params,
        )

        crop_shape = bg_slice.shape[1:]
        crop_to_bg = tuple(slice(0, int(size)) for size in crop_shape)
        anomaly_crop = anom[(slice(None), *crop_to_bg)]
        mask_crop = target_mask[crop_to_bg].astype(np.float32, copy=False)
        base_alpha = _soft_alpha(mask_crop, self.params, spatial_dims)
        support_mask = _support_mask(mask_crop, self.params, spatial_dims)

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


def _normalize_params(fusion_params):
    if fusion_params is None:
        return FusionConfiguration(Config()).fixed_params()
    if isinstance(fusion_params, FusionConfiguration):
        return fusion_params.fixed_params()
    return FusionConfiguration.from_value(fusion_params).fixed_params()


def _to_tensor(array, device):
    return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device).unsqueeze(0)


def _unpack_sample(sample):
    if isinstance(sample, dict):
        img = sample.get("img", sample.get("image"))
        seg = sample.get("seg", sample.get("mask", sample.get("ori_mask")))
        basename = sample.get("fname", sample.get("basename", "sample"))
        if img is None or seg is None:
            raise ValueError("Fusion backend training samples must contain image and mask data.")
        return img, seg, basename
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        basename = sample[2] if len(sample) >= 3 else "sample"
        return sample[0], sample[1], basename
    raise ValueError("Expected training sample as dict or tuple/list (img, seg, basename).")


def _spatial_label_mask(mask, spatial_dims):
    mask = np.asarray(mask)
    if mask.ndim == spatial_dims:
        return mask
    if mask.ndim == spatial_dims + 1:
        return np.max(mask, axis=0)
    raise ValueError(f"mask must have {spatial_dims} or {spatial_dims + 1} dims. Got {mask.shape}.")


def _bbox_slices(mask, *, margin, shape):
    coords = np.where(mask)
    return tuple(
        slice(
            max(0, int(axis.min()) - margin),
            min(int(shape[idx]), int(axis.max()) + margin + 1),
        )
        for idx, axis in enumerate(coords)
    )


def _channel_min(image):
    spatial_axes = tuple(range(1, image.ndim))
    mins = np.min(image, axis=spatial_axes, keepdims=True)
    return mins


def _pseudo_inpaint(target, mask, *, sigma):
    sigma_tuple = (0.0, *([max(float(sigma), 0.1)] * (target.ndim - 1)))
    blurred = scipy.ndimage.gaussian_filter(target, sigma=sigma_tuple)
    support = _support_mask(mask.astype(np.float32), {"residual_border_width": 2}, mask.ndim) > 0
    return np.where(support[None, ...], blurred, target).astype(np.float32, copy=False)


def _soft_alpha(mask, params, spatial_dims):
    mask = mask.astype(np.float32, copy=False)
    sigma = float(params.get("base_alpha_blur_sigma", 1.0))
    if sigma > 0:
        alpha = scipy.ndimage.gaussian_filter(mask, sigma=sigma)
    else:
        alpha = mask
    alpha = np.clip(alpha, 0.0, 1.0)
    max_value = float(np.max(alpha))
    if max_value > 0:
        alpha = alpha / max_value
    alpha = alpha * float(params.get("base_alpha", 0.85))
    support = _support_mask(mask, params, spatial_dims)
    alpha = np.where(support > 0, alpha, 0.0)
    return alpha.astype(np.float32, copy=False)


def _support_mask(mask, params, spatial_dims):
    iterations = int(params.get("residual_border_width", 0) or 0)
    support = mask > 0
    if iterations > 0 and np.any(support):
        structure = np.ones((3,) * spatial_dims, dtype=bool)
        support = binary_dilation(support, structure=structure, iterations=iterations)
    return support.astype(np.float32, copy=False)


def _safe_scale(array):
    scale = float(np.nanstd(array))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanmax(array) - np.nanmin(array))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return scale
