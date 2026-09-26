"""Paired source-to-target conditional ConvNeXt variational autoencoder."""

from __future__ import annotations

import math
from numbers import Real
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_sample_generator.generation.vae.base import HybridVAEBase
from hybrid_sample_generator.generation.vae.convnext.layers import ConvNeXtUNetEncoder
from hybrid_sample_generator.imaging.masks.encoding import to_one_hot
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator

from .configuration import Config
from .decoder import ConvNeXtSPADENoSkipDecoder


class PairedConditionalConvNeXtVAE(HybridVAEBase):
    """Encode a source once and decode it against a potentially different mask."""

    def __init__(
        self,
        cfg: Config,
        *,
        in_channels: int,
        num_anomaly_classes: int,
        spatial_dims: int,
    ) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
        if isinstance(cfg.identity_pair_probability, bool) or not isinstance(
            cfg.identity_pair_probability, Real
        ):
            raise TypeError("identity_pair_probability must be a real number.")
        if not 0.0 <= float(cfg.identity_pair_probability) <= 1.0:
            raise ValueError("identity_pair_probability must be in [0, 1].")
        self._validate_reconstruction_weights(cfg)
        self.cfg = cfg
        self.spatial_dims = int(spatial_dims)
        self.in_channels = int(in_channels)
        self.num_anomaly_classes = int(num_anomaly_classes)

        self.encoder = ConvNeXtUNetEncoder(
            in_channels=self.in_channels + self.num_anomaly_classes,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=0.0,
            skip_alpha=0.0,
            spatial_dims=spatial_dims,
        )
        self.decoder = ConvNeXtSPADENoSkipDecoder(
            out_channels=self.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_spade_blocks=cfg.n_spade_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            num_anomaly_classes=self.num_anomaly_classes,
            use_transpose_conv=cfg.use_transpose_conv,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            spatial_dims=spatial_dims,
        )
        self.fc_mu: Optional[nn.Linear] = None
        self.fc_logvar: Optional[nn.Linear] = None
        self.fc_decode: Optional[nn.Linear] = None
        self._latent_shape: Optional[tuple[int, ...]] = None

    def _ensure_fcs(self, latent_shape: tuple[int, ...], device: torch.device) -> None:
        if self._latent_shape == latent_shape and self.fc_mu is not None:
            return
        self._latent_shape = latent_shape
        flattened = int(self.cfg.z_channels * math.prod(latent_shape))
        self.fc_mu = nn.Linear(flattened, self.cfg.bottleneck_dim).to(device)
        self.fc_logvar = nn.Linear(flattened, self.cfg.bottleneck_dim).to(device)
        self.fc_decode = nn.Linear(self.cfg.bottleneck_dim, flattened).to(device)

    def forward(
        self,
        source_image: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if target_mask is None:
            target_mask = source_mask
        if target_image is None:
            target_image = source_image

        source_mask = to_one_hot(
            source_mask, self.num_anomaly_classes, spatial_dims=self.spatial_dims
        )
        target_mask = to_one_hot(
            target_mask, self.num_anomaly_classes, spatial_dims=self.spatial_dims
        )
        source_image = source_image.float()
        target_image = target_image.float()
        expected_ndim = self.spatial_dims + 2
        if any(
            value.ndim != expected_ndim
            for value in (source_image, source_mask, target_mask, target_image)
        ):
            raise ValueError(
                f"Expected batch, channel, and {self.spatial_dims} spatial dimensions."
            )
        if source_image.shape != target_image.shape:
            raise ValueError(
                f"Source shape {tuple(source_image.shape)} differs from target "
                f"shape {tuple(target_image.shape)}."
            )
        if source_image.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected C={self.in_channels}, got C={source_image.shape[1]}"
            )

        spatial_shape = tuple(source_image.shape[-self.spatial_dims :])
        source_pad, pad = self._pad_to_multiple(
            source_image, 2 ** self.cfg.n_levels
        )
        if sum(pad):
            source_mask_pad = F.pad(source_mask, pad, mode="constant", value=0.0)
            target_mask_pad = F.pad(target_mask, pad, mode="constant", value=0.0)
        else:
            source_mask_pad = source_mask
            target_mask_pad = target_mask

        encoded, _ = self.encoder(torch.cat([source_pad, source_mask_pad], dim=1))
        latent_shape = tuple(encoded.shape[-self.spatial_dims :])
        self._ensure_fcs(latent_shape, source_image.device)
        flattened = encoded.reshape(source_image.shape[0], -1)
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        z = self.reparameterize(mu, logvar)
        decoded = self.fc_decode(z).reshape(
            source_image.shape[0], self.cfg.z_channels, *latent_shape
        )
        reconstruction = self._crop_like(
            self.decoder(decoded, target_mask_pad), spatial_shape
        )
        return {
            "recon": reconstruction,
            "x_ref": target_image,
            "mu": mu,
            "logvar": logvar,
        }

    def _forward_args_from_batch(self, batch) -> tuple:
        if not isinstance(batch, dict):
            raise TypeError("Paired cVAE training expects a dictionary batch.")
        required = ("img", "ori_mask", "tgt_mask", "tgt_img")
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError(f"Paired training batch is missing keys: {missing}")
        return (
            batch["img"],
            batch["ori_mask"],
            batch["tgt_mask"],
            batch["tgt_img"],
        )

    @staticmethod
    def _reconstruction_mask_from_batch(batch) -> torch.Tensor:
        if isinstance(batch, dict) and "tgt_mask" in batch:
            value = batch["tgt_mask"]
            return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        raise ValueError("Paired cVAE training requires 'tgt_mask' in every batch.")

    def warmup(self, shape, device=None, dtype=None, config=None):
        if not isinstance(shape, (tuple, list)) or len(shape) != self.spatial_dims + 1:
            raise ValueError(
                f"shape must contain channels and {self.spatial_dims} spatial values, got {shape}"
            )
        shape = tuple(int(value) for value in shape)
        try:
            parameter = next(self.parameters())
            model_device, model_dtype = parameter.device, parameter.dtype
        except StopIteration:
            model_device, model_dtype = torch.device("cpu"), torch.float32
        device = model_device if device is None else torch.device(device)
        dtype = model_dtype if dtype is None else dtype
        was_training = self.training
        self.eval()
        with torch.no_grad():
            image = torch.zeros((1, *shape), device=device, dtype=dtype)
            mask = torch.zeros(
                (1, 1, *shape[1:]), device=device, dtype=torch.long
            )
            self(image, mask)
        self.train(was_training)
        return self

    def _generate_posterior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        original_mask=None,
        target_mask=None,
        *,
        n: int = 1,
        variation_strength: float = 0.5,
        device: Union[str, torch.device] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ):
        if n <= 0 or variation_strength < 0:
            raise ValueError("n must be positive and variation_strength non-negative.")
        if isinstance(sample, dict):
            source = torch.as_tensor(sample["img"]).float()
            if original_mask is None:
                original_mask = sample.get("ori_mask", sample.get("mask"))
            if target_mask is None:
                target_mask = sample.get("tgt_mask")
        else:
            source = torch.as_tensor(sample).float()
        if original_mask is None:
            raise ValueError("original_mask is required for conditional generation.")
        if target_mask is None and target_mask_generator is not None:
            target_mask = target_mask_generator.create_target_mask(
                original_mask=original_mask, conditional=True
            )
        if target_mask is None:
            target_mask = original_mask
        returned_mask = torch.as_tensor(target_mask)

        single = source.ndim == self.spatial_dims + 1
        if single:
            source = source.unsqueeze(0)
        elif source.ndim != self.spatial_dims + 2:
            raise ValueError(f"Unexpected source shape {tuple(source.shape)}")
        device = torch.device(device)
        source = source.clamp(0.0, 1.0).to(device) if clamp_01 else source.to(device)
        source_mask = to_one_hot(
            torch.as_tensor(original_mask).to(device),
            self.num_anomaly_classes,
            spatial_dims=self.spatial_dims,
        )
        target_one_hot = to_one_hot(
            returned_mask.to(device),
            self.num_anomaly_classes,
            spatial_dims=self.spatial_dims,
        )
        model = self.to(device).eval()
        with torch.no_grad():
            spatial_shape = tuple(source.shape[-self.spatial_dims :])
            source_pad, pad = self._pad_to_multiple(source, 2 ** self.cfg.n_levels)
            if sum(pad):
                source_mask = F.pad(source_mask, pad)
                target_one_hot = F.pad(target_one_hot, pad)
            encoded, _ = model.encoder(torch.cat([source_pad, source_mask], dim=1))
            latent_shape = tuple(encoded.shape[-self.spatial_dims :])
            model._ensure_fcs(latent_shape, device)
            mu = model.fc_mu(encoded.reshape(source.shape[0], -1))
            logvar = model.fc_logvar(encoded.reshape(source.shape[0], -1))
            std = torch.exp(0.5 * logvar)
            if variation_strength == 0.0:
                z = mu[:, None, :].expand(-1, n, -1).reshape(-1, mu.shape[-1])
            else:
                eps = torch.randn(
                    (source.shape[0], n, mu.shape[-1]), device=device, dtype=mu.dtype
                )
                z = (
                    mu[:, None, :] + variation_strength * std[:, None, :] * eps
                ).reshape(-1, mu.shape[-1])
            decoded = model.fc_decode(z).reshape(
                source.shape[0] * n, self.cfg.z_channels, *latent_shape
            )
            masks = target_one_hot.repeat_interleave(n, dim=0)
            result = self._crop_like(model.decoder(decoded, masks), spatial_shape)
            if clamp_01:
                result = result.clamp(0.0, 1.0)
            result = result.view(source.shape[0], n, self.in_channels, *spatial_shape)
            if single:
                result = result.squeeze(0)
                if n == 1:
                    result = result.squeeze(0)
        if return_torch:
            return result, returned_mask.to(result.device)
        return (
            result.detach().cpu().numpy().astype(np.float32, copy=False),
            returned_mask.cpu().numpy().astype(np.uint8, copy=False),
        )

    def _generate_prior(
        self,
        sample,
        *,
        variation_strength: float = 1.0,
        device: Union[str, torch.device] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ):
        if variation_strength < 0:
            raise ValueError("variation_strength must be non-negative.")
        if not isinstance(sample, dict):
            raise TypeError("Conditional prior generation requires a sample dictionary.")
        target_mask = sample.get("tgt_mask")
        original_mask = sample.get("ori_mask", sample.get("mask"))
        if target_mask is None and target_mask_generator is not None:
            target_mask = target_mask_generator.create_target_mask(
                original_mask=original_mask, conditional=True
            )
        if target_mask is None:
            target_mask = original_mask
        if target_mask is None:
            raise KeyError("Conditional prior generation requires a mask.")
        returned_mask = torch.as_tensor(target_mask)
        device = torch.device(device)
        one_hot = to_one_hot(
            returned_mask.to(device),
            self.num_anomaly_classes,
            spatial_dims=self.spatial_dims,
        )
        single = returned_mask.ndim <= self.spatial_dims + 1
        model = self.to(device).eval()
        with torch.no_grad():
            output_shape = tuple(one_hot.shape[2:])
            padded_mask, _ = self._pad_to_multiple(one_hot, 2 ** self.cfg.n_levels)
            latent_shape = tuple(
                size // (2 ** self.cfg.n_levels) for size in padded_mask.shape[2:]
            )
            model._ensure_fcs(latent_shape, device)
            z = torch.randn((one_hot.shape[0], self.cfg.bottleneck_dim), device=device)
            z *= variation_strength
            decoded = model.fc_decode(z).reshape(
                one_hot.shape[0], self.cfg.z_channels, *latent_shape
            )
            result = self._crop_like(model.decoder(decoded, padded_mask), output_shape)
            if clamp_01:
                result = result.clamp(0.0, 1.0)
            if single:
                result = result.squeeze(0)
        if return_torch:
            return result, returned_mask.to(result.device)
        return (
            result.detach().cpu().numpy().astype(np.float32, copy=False),
            returned_mask.cpu().numpy().astype(np.uint8, copy=False),
        )


__all__ = ["PairedConditionalConvNeXtVAE"]
