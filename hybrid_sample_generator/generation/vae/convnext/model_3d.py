"""ConvNeXt3D-U-Net VAE

Features:
- ConvNeXt3D blocks (depthwise conv + pointwise MLP)
- BatchNorm -> GroupNorm (more stable for 3D and small batch sizes)
- True U-Net skip connections (feature concatenation)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union, List
import math
import numpy as np

import torch
import torch.nn as nn

from hybrid_sample_generator.generation.vae.base import HybridVAEBase
from hybrid_sample_generator.imaging.masks.transforms import TransformGenerator



from .layers_3d import ConvNeXtUNetDecoder3D, ConvNeXtUNetEncoder3D

# -------------------------
# VAE 3D
# -------------------------

@dataclass
class Config:
    """Hyperparameters for ConvNeXtVAE3D.

    Kept identical to the original file for drop-in compatibility.

    Notes:
      - `use_transpose_conv` is still honored.
    """
    in_channels: int = None
    n_res_blocks: int = 8
    n_levels: int = 4
    z_channels: int = 250
    bottleneck_dim: int = 250
    recon_weight: float = 100.0
    beta_kl: float = 1.0
    beta_kl_start: float = 0.0
    beta_kl_max: float = 4.0
    beta_kl_warmup_start: float = 0
    beta_kl_warmup_epochs: int = 100
    free_bits: float = 0.0
    recon_loss: str = "smoothl1"  # 'smoothl1' or 'mse'
    recon_smoothl1_beta: float = 1.0
    use_transpose_conv: bool = True
    fg_weight: float = 1.0
    fg_threshold: float = 0.0

    # Probability for dropping encoder skip features during training (prevents latent bypass in U-Net VAEs)
    # 0.0 disables skip dropout. Typical values: 0.1 - 0.4
    skip_dropout_p: float = 0.0
    # Optional per-resolution skip dropout values in encoder order: [highest resolution, ..., deepest].
    # If set, this overrides skip_dropout_p for individual skip levels.
    skip_dropout_ps: Optional[List[float]] = None

    # Skip gating factor: scales skip features before concatenation in the decoder.
    # 1.0 disables gating (default). Typical values for encouraging latent usage: 0.2 - 0.6
    skip_alpha: float = 1.0


class ConvNeXtVAE3D(HybridVAEBase):
    """3D ConvNeXt-U-Net VAE.

    Expected input:
      - x: (B, C, D, H, W), float (continuous intensities).

    Forward output is a dict:
      - recon: reconstructed x (B,C,D,H,W)
      - mu: mean vector (B, bottleneck_dim)
      - logvar: log-variance vector (B, bottleneck_dim)
      - x_ref: reference input used for reconstruction loss (cropped/padded version)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels

        # Encoder returns (h, skips)
        self.encoder = ConvNeXtUNetEncoder3D(
            in_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
        )

        # Decoder reconstructs from latent feature map; skips are set each forward
        self.decoder = ConvNeXtUNetDecoder3D(
            out_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_transpose_conv=cfg.use_transpose_conv,
            skip_dropout_p=cfg.skip_dropout_p,
            skip_dropout_ps=cfg.skip_dropout_ps,
            skip_alpha=cfg.skip_alpha,
        )

        # Lazy FC layers (depend on latent spatial size)
        self.fc_mu: Optional[nn.Linear] = None
        self.fc_logvar: Optional[nn.Linear] = None
        self.fc_decode: Optional[nn.Linear] = None
        self._latent_dhw: Optional[Tuple[int, int, int]] = None

    def _ensure_fcs(self, latent_dhw: Tuple[int, int, int], device: torch.device):
        """Lazily create bottleneck fully-connected layers when latent spatial size changes."""
        if self._latent_dhw == latent_dhw and self.fc_mu is not None:
            return

        self._latent_dhw = latent_dhw
        flat = int(self.cfg.z_channels * math.prod(latent_dhw))

        self.fc_mu = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_logvar = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_decode = nn.Linear(self.cfg.bottleneck_dim, flat).to(device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder -> bottleneck -> decoder."""
        if x.ndim != 5:
            raise ValueError(f"Expected (B,C,D,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[1]}")

        x = x.float()
        device = x.device
        B = x.shape[0]
        ref_dhw = tuple(x.shape[-3:])

        # Pad so D/H/W divisible by 2**n_levels
        multiple = 2 ** self.cfg.n_levels
        x_pad, pad = self._pad_to_multiple(x, multiple)

        # Encode -> (latent feature map, skips)
        h, skips = self.encoder(x_pad)
        latent_dhw = tuple(h.shape[-3:])

        # Ensure FC layers
        self._ensure_fcs(latent_dhw, device)

        # Flatten -> mu/logvar
        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        # Sample bottleneck
        z = self.reparameterize(mu, logvar)

        # Decode
        h_dec = self.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_dhw)
        self.decoder.set_skips(skips)
        recon = self.decoder(h_dec)

        # Crop recon back to original spatial size
        recon = self._crop_like(recon, ref_dhw)
        x_ref = self._crop_like(x_pad, ref_dhw) if sum(pad) else x

        return {"recon": recon, "mu": mu, "logvar": logvar, "x_ref": x_ref}

    def _extract_x(self, batch) -> torch.Tensor:
        """Extract the input tensor x from a batch (unchanged from original)."""
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
        variation_strength: float = 0.8,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Generate *n* slightly varied variants around a given sample.

        This performs *posterior sampling*:
            z = mu + variation_strength * sigma * eps,  eps ~ N(0, I)

        Meaning of the parameters:
          - n: how many variants to generate per input sample.
          - variation_strength: strength of the variation.
               variation_strength=0.0 -> deterministic reconstruction (uses mu only)
               variation_strength~0.2-0.5 -> small variations (recommended)
               variation_strength>=1.0 -> large variations (can drift away from the input)

        Inputs
        ------
        sample:
            Sample dict containing "img", or raw (C,D,H,W) / (B,C,D,H,W).

        Outputs
        -------
        If input is (C,D,H,W):
            returns (n,C,D,H,W)
        If input is (B,C,D,H,W):
            returns (B,n,C,D,H,W)
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
        if x.ndim == 4:
            x = x.unsqueeze(0)  # (1,C,D,H,W)
            single = True
        elif x.ndim == 5:
            pass
        else:
            raise ValueError(f"Expected (C,D,H,W) or (B,C,D,H,W), got {tuple(x.shape)}")

        if clamp_01:
            x = x.clamp(0.0, 1.0)

        x = x.to(device)

        with torch.no_grad():
            # --- same preprocessing as forward() ---
            ref_dhw = tuple(x.shape[-3:])
            multiple = 2 ** self.cfg.n_levels
            x_pad, pad = self._pad_to_multiple(x, multiple)

            # Encode once
            h, skips = model.encoder(x_pad)
            latent_dhw = tuple(h.shape[-3:])
            model._ensure_fcs(latent_dhw, device)

            B = x.shape[0]
            h_flat = h.reshape(B, -1)
            mu = model.fc_mu(h_flat)
            logvar = model.fc_logvar(h_flat)
            std = torch.exp(0.5 * logvar)

            # Sample n variants per item
            # Shape: (B*n, bottleneck_dim)
            if variation_strength == 0.0:
                z = mu.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
            else:
                eps = torch.randn((B, n, mu.shape[-1]), device=device, dtype=mu.dtype)
                z = (mu.unsqueeze(1) + (variation_strength * std).unsqueeze(1) * eps).reshape(B * n, -1)

            # Decode in one big batch
            h_dec = model.fc_decode(z).reshape(B * n, self.cfg.z_channels, *latent_dhw)

            # Repeat skips to match B*n
            rep_skips: List[torch.Tensor] = []
            for sk in skips:
                rep_skips.append(sk.repeat_interleave(n, dim=0))
            model.decoder.set_skips(rep_skips)

            recon = model.decoder(h_dec)
            recon = self._crop_like(recon, ref_dhw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            # Reshape back to (B,n,C,D,H,W)
            recon = recon.view(B, n, self.in_channels, *ref_dhw)

            if single:
                recon = recon.squeeze(0)  # (n,C,D,H,W)
                recon = recon.squeeze(0)

        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()

        if return_torch:
            return recon, target_mask_generator.create_target_mask(synth_anomaly_image=recon)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, target_mask_generator.create_target_mask(synth_anomaly_image=recon_np)

    def warmup(self, shape, device=None, dtype=None, config=None):
        """Warm up the model to initialize lazy FC layers (unchanged API)."""
        if not (isinstance(shape, (tuple, list)) and len(shape) == 4):
            raise ValueError(f"shape must be (C,D,H,W), got: {shape}")

        C, D, H, W = map(int, shape)
        if min(C, D, H, W) <= 0:
            raise ValueError(f"All dimensions must be > 0, got: {shape}")

        try:
            p = next(self.parameters())
            model_device = p.device
            model_dtype = p.dtype
        except StopIteration:
            model_device = torch.device("cpu")
            model_dtype = torch.float32

        if device is None:
            device = model_device
        else:
            device = torch.device(device)

        if dtype is None:
            dtype = model_dtype

        was_training = self.training
        self.eval()

        with torch.no_grad():
            x = torch.zeros((1, C, D, H, W), device=device, dtype=dtype)
            _ = self(x)

        if was_training:
            self.train()

        return self
    
    def _generate_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor, None] = None,
        *,
        out_dhw: tuple[int, int, int] | None = None,
        variation_strength: float = 0.5,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Generate ONE synthetic 3D sample via *prior sampling* (no input sample required).

        Samples:
            z ~ N(0, I)  (scaled by variation_strength), then decode to 3D volume space.

        Parameters:
        - out_dhw: output (D, H, W). If None, tries cfg.sample_dhw or cfg.image_dhw, else defaults to (64, 64, 64).
        - variation_strength: prior diversity strength (1.0 is standard; <1.0 more conservative; >1.0 more diverse).
        - clamp_01: clamp outputs to [0,1].
        - return_torch: return torch.Tensor instead of np.ndarray.

        Output:
        - (C, D, H, W)
        """
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        # pick output size
        if out_dhw is None and sample is not None:
            out_dhw = tuple(self._extract_x(sample).shape[-3:])
        if not (isinstance(out_dhw, (tuple, list)) and len(out_dhw) == 3):
            raise ValueError(f"out_dhw must be (D, H, W), got {out_dhw}")
        
        D, H, W = int(out_dhw[0]), int(out_dhw[1]), int(out_dhw[2])
        if D <= 0 or H <= 0 or W <= 0:
            raise ValueError(f"out_dhw must be positive, got {out_dhw}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        # Ensure decoder doesn't expect encoder skips (we have none for pure prior sampling)

        model.decoder.set_skips(None)


        # Compute latent spatial size (assuming 2x downsample per level)
        down = 2 ** int(self.cfg.n_levels)

        # Pad to be divisible by down (so latent grid is integer)
        pad_d = (down - (D % down)) % down
        pad_h = (down - (H % down)) % down
        pad_w = (down - (W % down)) % down
        
        D_pad, H_pad, W_pad = D + pad_d, H + pad_h, W + pad_w
        latent_dhw = (D_pad // down, H_pad // down, W_pad // down)

        # Determine latent vector dim for fc_decode (matches your fc_mu/fc_logvar output)
        z_dim = int(self.cfg.bottleneck_dim)

        with torch.no_grad():
            model._ensure_fcs(latent_dhw, device)

            # Prior sampling: z ~ N(0, I)
            if variation_strength == 0.0:
                z = torch.zeros((1, z_dim), device=device)
            else:
                z = torch.randn((1, z_dim), device=device) * float(variation_strength)

            # Map z -> decoder feature map and decode
            h_dec = model.fc_decode(z).reshape(1, int(self.cfg.z_channels), *latent_dhw)
            recon = model.decoder(h_dec)  # (1, C, D_pad, H_pad, W_pad) typically

            # Crop back to requested size and drop batch dim
            recon = recon[..., :D, :H, :W].squeeze(0)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()

        if return_torch:
            return recon, target_mask_generator.create_target_mask(synth_anomaly_image=recon)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, target_mask_generator.create_target_mask(synth_anomaly_image=recon_np)
