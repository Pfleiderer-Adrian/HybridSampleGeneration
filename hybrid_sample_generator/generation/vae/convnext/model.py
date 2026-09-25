"""Dimension-independent ConvNeXt U-Net variational autoencoder."""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Union
import math
import numpy as np

import torch
import torch.nn as nn

from hybrid_sample_generator.generation.vae.base import HybridVAEBase
from .configuration import Config
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


from .layers import ConvNeXtUNetDecoder, ConvNeXtUNetEncoder

# -------------------------
# VAE 2D
# -------------------------



class ConvNeXtVAE(HybridVAEBase):
    """ConvNeXt U-Net VAE.

    Expected input:
      - x: (B, C, H, W)

    Forward output dict:
      - recon: reconstructed x (B,C,H,W)
      - mu: mean vector (B, bottleneck_dim)
      - logvar: log-variance vector (B, bottleneck_dim)
      - x_ref: reference input used for reconstruction loss (cropped/padded version)
    """

    def __init__(self, cfg: Config, *, in_channels: int, spatial_dims: int):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
        self.spatial_dims = spatial_dims
        self.cfg = cfg
        self._validate_latent_recon_config(cfg)
        self._validate_reconstruction_weights(cfg)
        self.in_channels = int(in_channels)

        self.encoder = ConvNeXtUNetEncoder(
            in_channels=self.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=cfg.skip_dropout_p,
            skip_alpha=cfg.skip_alpha,
            spatial_dims=spatial_dims,
        )

        self.decoder = ConvNeXtUNetDecoder(
            out_channels=self.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_transpose_conv=cfg.use_transpose_conv,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=cfg.skip_dropout_p,
            skip_dropout_ps=cfg.skip_dropout_ps,
            skip_alpha=cfg.skip_alpha,
            skip_alphas=cfg.skip_alphas,
            spatial_dims=spatial_dims,
        )

        # Lazy FC layers (depend on latent spatial size)
        self.fc_mu: Optional[nn.Linear] = None
        self.fc_logvar: Optional[nn.Linear] = None
        self.fc_decode: Optional[nn.Linear] = None
        self._latent_shape: Optional[Tuple[int, ...]] = None

    def _ensure_fcs(self, latent_shape: Tuple[int, ...], device: torch.device):
        """Lazily create bottleneck fully-connected layers when latent size changes."""
        if self._latent_shape == latent_shape and self.fc_mu is not None:
            return

        self._latent_shape = latent_shape
        flat = int(self.cfg.z_channels * math.prod(latent_shape))

        self.fc_mu = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_logvar = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_decode = nn.Linear(self.cfg.bottleneck_dim, flat).to(device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder -> bottleneck -> decoder."""
        if x.ndim != self.spatial_dims + 2:
            raise ValueError(f"Expected a batch, channel, and {self.spatial_dims} spatial dimensions, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[1]}")

        x = x.float()
        device = x.device
        B = x.shape[0]
        spatial_shape = tuple(x.shape[-self.spatial_dims:])

        multiple = 2 ** self.cfg.n_levels
        x_pad, pad = self._pad_to_multiple(x, multiple)

        h, skips = self.encoder(x_pad)
        latent_shape = tuple(h.shape[-self.spatial_dims:])
        self._ensure_fcs(latent_shape, device)

        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        z = self.reparameterize(mu, logvar)

        h_dec = self.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_shape)
        self.decoder.set_skips(skips)
        recon = self.decoder(h_dec)

        result = {"mu": mu, "logvar": logvar}
        if self.cfg.latent_recon_weight > 0.0:
            latent_target = (
                mu.detach() + self.cfg.latent_recon_noise_scale * torch.randn_like(mu)
            )
            cycle_h_dec = self.fc_decode(latent_target).reshape(
                B, self.cfg.z_channels, *latent_shape
            )
            cycle_recon = self.decoder(cycle_h_dec)
            if self.training and self.cfg.latent_recon_image_noise_std > 0.0:
                cycle_recon = cycle_recon + (
                    self.cfg.latent_recon_image_noise_std * torch.randn_like(cycle_recon)
                )
            cycle_h, _ = self.encoder(cycle_recon)
            cycle_mu = self.fc_mu(cycle_h.reshape(B, -1))
            result.update({"latent_recon": cycle_mu, "latent_target": latent_target})

        recon = self._crop_like(recon, spatial_shape)
        x_ref = self._crop_like(x_pad, spatial_shape) if sum(pad) else x

        result.update({"recon": recon, "x_ref": x_ref})
        return result

    def _extract_x(self, batch) -> torch.Tensor:
        """Extract the input tensor x from a batch (kept compatible with the template)."""
        if isinstance(batch, list) and len(batch) == 2:
            if isinstance(batch[0], torch.Tensor):
                return batch[0]

        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, np.ndarray):
            return torch.as_tensor(batch)

        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            x = batch[0]
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)

        if isinstance(batch, dict):
            for key in ("img", "x", "image", "inputs"):
                if key in batch:
                    v = batch[key]
                    if isinstance(v, torch.Tensor):
                        return v
                    return torch.as_tensor(v)

        raise TypeError(f"Unknown batch type: {type(batch)}")

    def _generate_posterior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        *,
        n: int = 1,
        variation_strength: float = 0.5,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Generate *n* slightly varied variants around a given sample.

        This performs posterior sampling:
            z = mu + variation_strength * sigma * eps,  eps ~ N(0, I)

        Parameters:
          - n: number of variants per input sample.
          - variation_strength: strength of the variation.
               variation_strength=0.0 -> deterministic reconstruction (uses mu only)
               variation_strength~0.2-0.5 -> small variations (recommended)
               variation_strength>=1.0 -> large variations (can drift away)

        Input:
          - sample: dict containing "img", or raw (C,H,W) / (B,C,H,W)

        Output:
          - if input is (C,H,W): (n,C,H,W)
          - if input is (B,C,H,W): (B,n,C,H,W)
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        x = self._extract_x(sample)

        x = x.float()
        single = False
        if x.ndim == self.spatial_dims + 1:
            x = x.unsqueeze(0)  # (1,C,H,W)
            single = True
        elif x.ndim == self.spatial_dims + 2:
            pass
        else:
            raise ValueError(f"Expected channel-first input with or without a batch, got {tuple(x.shape)}")

        if clamp_01:
            x = x.clamp(0.0, 1.0)

        x = x.to(device)
        #model.train()      # wichtig!

        with torch.no_grad():
            spatial_shape = tuple(x.shape[-self.spatial_dims:])
            multiple = 2 ** self.cfg.n_levels
            x_pad, pad = self._pad_to_multiple(x, multiple)

            h, skips = model.encoder(x_pad)
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

            
            recon = model.decoder(h_dec)
            recon = self._crop_like(recon, spatial_shape)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            recon = recon.view(B, n, self.in_channels, *spatial_shape)

            if single:
                recon = recon.squeeze(0) 
                recon = recon.squeeze(0)  # (n,C,H,W)
 # (n,C,H,W)

        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()

        if return_torch:
            return recon, target_mask_generator.create_target_mask(synth_anomaly_image=recon)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, target_mask_generator.create_target_mask(synth_anomaly_image=recon_np)


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
        with torch.no_grad():
            self(torch.zeros((1, *shape), device=device, dtype=dtype))
        if was_training:
            self.train()
        return self

    def _generate_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor, None] = None,
        *,
        output_shape: tuple[int, ...] | None = None,
        variation_strength: float = 1.0,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Generate one sample by decoding a standard-normal latent vector."""
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")
        if output_shape is None and sample is not None:
            output_shape = tuple(self._extract_x(sample).shape[-self.spatial_dims:])
        if not isinstance(output_shape, (tuple, list)) or len(output_shape) != self.spatial_dims:
            raise ValueError(
                f"output_shape must contain {self.spatial_dims} values, got {output_shape}"
            )
        output_shape = tuple(int(value) for value in output_shape)
        if min(output_shape) <= 0:
            raise ValueError(f"output_shape must be positive, got {output_shape}")
        device = torch.device(device)
        model = self.to(device)
        model.eval()
        model.decoder.set_skips(None)
        down = 2 ** int(self.cfg.n_levels)
        padded_shape = tuple(size + (down - size % down) % down for size in output_shape)
        latent_shape = tuple(size // down for size in padded_shape)
        with torch.no_grad():
            model._ensure_fcs(latent_shape, device)
            if variation_strength == 0.0:
                z = torch.zeros((1, int(self.cfg.bottleneck_dim)), device=device)
            else:
                z = torch.randn((1, int(self.cfg.bottleneck_dim)), device=device)
                z = z * float(variation_strength)
            decoded = model.fc_decode(z).reshape(
                1, int(self.cfg.z_channels), *latent_shape
            )
            recon = self._crop_like(model.decoder(decoded), output_shape).squeeze(0)
            if clamp_01:
                recon = recon.clamp(0.0, 1.0)
        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()
        if return_torch:
            return recon, target_mask_generator.create_target_mask(
                synth_anomaly_image=recon
            )
        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, target_mask_generator.create_target_mask(
            synth_anomaly_image=recon_np
        )
