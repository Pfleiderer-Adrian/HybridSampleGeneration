"""Dimension-independent conditional ConvNeXt variational autoencoder."""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Union
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from hybrid_sample_generator.generation.vae.base import HybridVAEBase
from .configuration import Config
from hybrid_sample_generator.imaging.masks.encoding import to_one_hot
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


from hybrid_sample_generator.generation.vae.convnext.layers import ConvNeXtUNetEncoder
from .spade import ConvNeXtSPADEUNetDecoder

# -------------------------
# VAE 2D conditional
# -------------------------



class ConditionalConvNeXtVAE(HybridVAEBase):
    """Conditional ConvNeXt U-Net VAE with SPADE for conditional generation."""

    def __init__(
        self,
        cfg: Config,
        *,
        in_channels: int,
        num_anomaly_classes: int,
        spatial_dims: int,
    ):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
        self.spatial_dims = spatial_dims
        self.cfg = cfg
        self._validate_latent_recon_config(cfg)
        self.in_channels = int(in_channels)
        self.num_anomaly_classes = int(num_anomaly_classes)

        # encoder gets real mask as additional input (concatenated)
        enc_in_channels = self.in_channels + self.num_anomaly_classes

        self.encoder = ConvNeXtUNetEncoder(
            in_channels=enc_in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=cfg.skip_dropout_p,
            skip_alpha=cfg.skip_alpha,
            spatial_dims=spatial_dims,
        )

        self.decoder = ConvNeXtSPADEUNetDecoder(
            out_channels=self.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_spade_blocks=cfg.n_spade_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            num_anomaly_classes=self.num_anomaly_classes,
            use_transpose_conv=cfg.use_transpose_conv,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=cfg.skip_dropout_p,
            skip_dropout_ps=cfg.skip_dropout_ps,
            skip_alpha=cfg.skip_alpha,
            skip_alphas=cfg.skip_alphas,
            spatial_dims=spatial_dims,
        )

        self.fc_mu: Optional[nn.Linear] = None
        self.fc_logvar: Optional[nn.Linear] = None
        self.fc_decode: Optional[nn.Linear] = None
        self._latent_shape: Optional[Tuple[int, ...]] = None

    def _ensure_fcs(self, latent_shape: Tuple[int, ...], device: torch.device):
        if self._latent_shape == latent_shape and self.fc_mu is not None:
            return

        self._latent_shape = latent_shape
        flat = int(self.cfg.z_channels * math.prod(latent_shape))

        self.fc_mu = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_logvar = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_decode = nn.Linear(self.cfg.bottleneck_dim, flat).to(device)

    def forward(self, x: torch.Tensor, ori_mask: torch.Tensor, tgt_mask: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        if tgt_mask is None:
            tgt_mask = ori_mask
            
        ori_mask = to_one_hot(ori_mask, self.num_anomaly_classes, spatial_dims=self.spatial_dims)
        tgt_mask = to_one_hot(tgt_mask, self.num_anomaly_classes, spatial_dims=self.spatial_dims)
        
        if any(value.ndim != self.spatial_dims + 2 for value in (x, ori_mask, tgt_mask)):
            raise ValueError(f"Expected a batch, channel, and {self.spatial_dims} spatial dimensions, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[1]}")

        x = x.float()
        device = x.device
        B = x.shape[0]
        spatial_shape = tuple(x.shape[-self.spatial_dims:])

        multiple = 2 ** self.cfg.n_levels
        x_pad, pad = self._pad_to_multiple(x, multiple)
        if sum(pad) > 0:
            ori_mask_pad = F.pad(ori_mask, pad, mode="constant", value=0.0)
            tgt_mask_pad = F.pad(tgt_mask, pad, mode="constant", value=0.0)
        else:
            ori_mask_pad = ori_mask
            tgt_mask_pad = tgt_mask

        # Encode -> (latent feature map, skips)
        enc_in = torch.cat([x_pad, ori_mask_pad], dim=1)
        h, skips = self.encoder(enc_in)
        latent_shape = tuple(h.shape[-self.spatial_dims:])

        self._ensure_fcs(latent_shape, device)

        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        z = self.reparameterize(mu, logvar)

        # Decode
        h_dec = self.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_shape)
        self.decoder.set_skips(skips)
        
        recon = self.decoder(h_dec, tgt_mask_pad)

        result = {"mu": mu, "logvar": logvar}
        if self.cfg.latent_recon_weight > 0.0:
            latent_target = (
                mu.detach() + self.cfg.latent_recon_noise_scale * torch.randn_like(mu)
            )
            cycle_h_dec = self.fc_decode(latent_target).reshape(
                B, self.cfg.z_channels, *latent_shape
            )
            cycle_recon = self.decoder(cycle_h_dec, tgt_mask_pad)
            if self.training and self.cfg.latent_recon_image_noise_std > 0.0:
                cycle_recon = cycle_recon + (
                    self.cfg.latent_recon_image_noise_std * torch.randn_like(cycle_recon)
                )
            cycle_input = torch.cat([cycle_recon, tgt_mask_pad], dim=1)
            cycle_h, _ = self.encoder(cycle_input)
            cycle_mu = self.fc_mu(cycle_h.reshape(B, -1))
            result.update({"latent_recon": cycle_mu, "latent_target": latent_target})

        recon = self._crop_like(recon, spatial_shape)
        x_ref = self._crop_like(x_pad, spatial_shape) if sum(pad) else x

        result.update({"recon": recon, "x_ref": x_ref})
        return result

    def _extract_inputs(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract x and ori_mask from batch; tgt_mask is optional for generation-only use."""
        if isinstance(batch, dict):
            x = batch.get("img", batch.get("x"))
            ori_mask = batch.get("ori_mask", batch.get("mask"))
            tgt_mask = batch.get("tgt_mask")
            if x is not None and ori_mask is not None:
                return (
                    torch.as_tensor(x),
                    torch.as_tensor(ori_mask),
                    torch.as_tensor(tgt_mask) if tgt_mask is not None else None,
                )
            else:
                raise ValueError(
                f"Dataloader returned a dict, but expected keys are missing. "
                f"Found keys: {list(batch.keys())}. "
                f"Expected combinations of ('img' or 'x') and ('mask' or 'ori_mask')."
            )

        if isinstance(batch, (tuple, list)) and len(batch) >= 3:
            return torch.as_tensor(batch[0]), torch.as_tensor(batch[1]), torch.as_tensor(batch[2])
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return torch.as_tensor(batch[0]), torch.as_tensor(batch[1]), None

        raise TypeError(f"Unknown batch type: {type(batch)}")

    def _forward_args_from_batch(self, batch) -> tuple:
        x, ori_mask, _ = self._extract_inputs(batch)
        return x, ori_mask, ori_mask

    def _generate_posterior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        original_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        *,
        n: int = 1,
        variation_strength: float = 0.5,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        if isinstance(sample, dict):
            if "img" not in sample:
                raise KeyError("Conditional sample dict must contain 'img'.")
            x = torch.as_tensor(sample["img"]).float()
            if original_mask is None:
                original_mask = sample.get("ori_mask", sample.get("mask"))
            if target_mask is None:
                target_mask = sample.get("tgt_mask")
        else:
            x = torch.as_tensor(sample).float()

        if original_mask is None:
            raise ValueError("original_mask is required for conditional generation.")
        
        ori_mask = torch.as_tensor(original_mask)
        if target_mask is None and target_mask_generator is not None:
            target_mask = target_mask_generator.create_target_mask(original_mask=original_mask, conditional=True)

        if target_mask is None:
            tgt_mask = ori_mask
        else:
            tgt_mask = torch.as_tensor(target_mask)
        tgt_mask_return = tgt_mask

        single = False
        if x.ndim == self.spatial_dims + 1:
            x = x.unsqueeze(0)  # (1,C,H,W)
            single = True
        elif x.ndim != self.spatial_dims + 2:
            raise ValueError(f"Expected channel-first input with or without a batch, got {tuple(x.shape)}")

        if clamp_01:
            x = x.clamp(0.0, 1.0)

        x = x.to(device)
        
        ori_mask = to_one_hot(ori_mask.to(device), self.num_anomaly_classes, spatial_dims=self.spatial_dims)
        tgt_mask = to_one_hot(tgt_mask.to(device), self.num_anomaly_classes, spatial_dims=self.spatial_dims)

        with torch.no_grad():
            spatial_shape = tuple(x.shape[-self.spatial_dims:])
            multiple = 2 ** self.cfg.n_levels
            x_pad, pad = self._pad_to_multiple(x, multiple)
            if sum(pad) > 0:
                ori_mask_pad = F.pad(ori_mask, pad, mode="constant", value=0.0)
                tgt_mask_pad = F.pad(tgt_mask, pad, mode="constant", value=0.0)
            else:
                ori_mask_pad = ori_mask
                tgt_mask_pad = tgt_mask

            enc_in = torch.cat([x_pad, ori_mask_pad], dim=1)
            h, skips = model.encoder(enc_in)
            latent_shape = tuple(h.shape[-self.spatial_dims:])
            model._ensure_fcs(latent_shape, device)

            B = x.shape[0]
            h_flat = h.reshape(B, -1)
            mu = model.fc_mu(h_flat)
            logvar = model.fc_logvar(h_flat)
            std = torch.exp(0.5 * logvar)

            if variation_strength == 0.0:
                z = mu.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
            else:
                eps = torch.randn((B, n, mu.shape[-1]), device=device, dtype=mu.dtype)
                z = (mu.unsqueeze(1) + (variation_strength * std).unsqueeze(1) * eps).reshape(B * n, -1)

            h_dec = model.fc_decode(z).reshape(B * n, self.cfg.z_channels, *latent_shape)

            if not any(self.decoder.skip_alphas):
                model.decoder.set_skips(None)
            else:
                # The decoder applies the configured per-level scales. Pass raw skips
                # so posterior generation uses the same scaling as training.
                rep_skips = [sk.repeat_interleave(n, dim=0) for sk in skips]
                model.decoder.set_skips(rep_skips)

            tgt_mask_pad_rep = tgt_mask_pad.repeat_interleave(n, dim=0)
            recon = model.decoder(h_dec, tgt_mask_pad_rep)
            recon = self._crop_like(recon, spatial_shape)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            recon = recon.view(B, n, self.in_channels, *spatial_shape)

            if single:
                recon = recon.squeeze(0)
                recon = recon.squeeze(0)

        if return_torch:
            return recon, tgt_mask_return.to(recon.device)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        tgt_mask_np = tgt_mask_return.cpu().numpy().astype(np.uint8, copy=False)
        return recon_np, tgt_mask_np

    def warmup(self, shape, device=None, dtype=None, config=None):
        """Initialize shape-dependent fully connected layers."""
        if not isinstance(shape, (tuple, list)) or len(shape) != self.spatial_dims + 1:
            raise ValueError(
                f"shape must contain channels and {self.spatial_dims} spatial values, got {shape}"
            )
        shape = tuple(int(value) for value in shape)
        if min(shape) <= 0:
            raise ValueError(f"All dimensions must be > 0, got: {shape}")
        try:
            parameter = next(self.parameters())
            model_device, model_dtype = parameter.device, parameter.dtype
        except StopIteration:
            model_device, model_dtype = torch.device("cpu"), torch.float32
        device = model_device if device is None else torch.device(device)
        dtype = model_dtype if dtype is None else dtype
        was_training = self.training
        self.eval()
        spatial_shape = shape[1:]
        with torch.no_grad():
            image = torch.zeros((1, *shape), device=device, dtype=dtype)
            mask = torch.zeros((1, 1, *spatial_shape), device=device, dtype=torch.long)
            self(image, mask)
        if was_training:
            self.train()
        return self

    def _generate_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        *,
        variation_strength: float = 1.0,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Generate one sample from a target mask and a standard-normal latent vector."""
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")
        device = torch.device(device)
        model = self.to(device)
        model.eval()
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
            raise KeyError("Conditional prior sample must contain 'tgt_mask' or 'ori_mask'.")

        target_mask = torch.as_tensor(target_mask)
        returned_mask = target_mask
        single = target_mask.ndim <= self.spatial_dims + 1
        target_one_hot = to_one_hot(
            target_mask.to(device),
            self.num_anomaly_classes,
            spatial_dims=self.spatial_dims,
        )
        model.decoder.set_skips(None)
        with torch.no_grad():
            output_shape = tuple(target_one_hot.shape[2:])
            multiple = 2 ** self.cfg.n_levels
            padded_mask, _ = self._pad_to_multiple(target_one_hot, multiple)
            latent_shape = tuple(size // multiple for size in padded_mask.shape[2:])
            model._ensure_fcs(latent_shape, device)
            batch_size = target_one_hot.shape[0]
            if variation_strength == 0.0:
                z = torch.zeros((batch_size, int(self.cfg.bottleneck_dim)), device=device)
            else:
                z = torch.randn((batch_size, int(self.cfg.bottleneck_dim)), device=device)
                z = z * float(variation_strength)
            decoded = model.fc_decode(z).reshape(
                batch_size, int(self.cfg.z_channels), *latent_shape
            )
            recon = self._crop_like(model.decoder(decoded, padded_mask), output_shape)
            if clamp_01:
                recon = recon.clamp(0.0, 1.0)
            if single:
                recon = recon.squeeze(0)
        if return_torch:
            return recon, returned_mask.to(recon.device)
        return (
            recon.detach().cpu().numpy().astype(np.float32, copy=False),
            returned_mask.cpu().numpy().astype(np.uint8, copy=False),
        )
