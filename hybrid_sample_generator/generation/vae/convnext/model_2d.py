"""ConvNeXt2D-U-Net VAE

2D variant of the ConvNeXt3D-U-Net VAE

Features:
- ConvNeXt2D blocks (depthwise conv + pointwise MLP)
- GroupNorm instead of BatchNorm (stable for small batch sizes)
- True U-Net skip connections (feature concatenation)
- Posterior sampling for *lightly varied* variants around a given input

Shapes:
- Input x: (B, C, H, W)
- Output recon: (B, C, H, W)


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, Union, List
import math
import numpy as np

import torch
import torch.nn as nn

from hybrid_sample_generator.generation.vae.base import HybridVAEBase
from hybrid_sample_generator.imaging.masks.transforms import TransformGenerator


from .layers_2d import (
    ConvNeXtBlock2D,
    ConvNeXtUNetDecoder2D,
    ConvNeXtUNetEncoder2D,
    DropPath,
    _best_gn_groups,
    _upsample_block2d,
)

# -------------------------
# VAE 2D
# -------------------------

@dataclass
class Config:
    """Hyperparameters for ConvNeXtVAE2D.

    Notes:
      - use_transpose_conv is honored.
    """
    in_channels:int = None
    n_res_blocks: int = 8
    n_levels: int = 4
    z_channels: int = 250
    bottleneck_dim: int = 250
    recon_weight: float = 100.0
    beta_kl: float = 1.0
    beta_kl_start: float = 0.0
    beta_kl_max: float = 0.03
    beta_kl_warmup_start: int = 20
    beta_kl_warmup_epochs: int = 30
    free_bits: float = 0.0

    recon_loss: str = "smoothl1"  # 'smoothl1' or 'mse'
    recon_smoothl1_beta: float = 1.0
    use_transpose_conv: bool = True
    fg_weight: float = 1.0
    fg_threshold: float = 0.0

    # Regularization
    drop_path_rate: float = 0.10  # Stochastic depth max rate (0.0 disables)
    dropout: float = 0.05         # Dropout inside MLP (0.0 disables)

    # Skip regularization (helps force latent usage)
    skip_dropout_p: float = 0.0  # Drop entire skip-tensors per sample during training (0.0 disables)
    # Optional per-resolution skip dropout values in encoder order: [highest resolution, ..., deepest].
    # If set, this overrides skip_dropout_p for individual skip levels.
    skip_dropout_ps: Optional[List[float]] = None
    skip_alpha: float = 1.0      # Scale skips (0.0 disables skips, 0.2 keeps small guidance)


class ConvNeXtVAE2D(HybridVAEBase):
    """2D ConvNeXt-U-Net VAE.

    Expected input:
      - x: (B, C, H, W)

    Forward output dict:
      - recon: reconstructed x (B,C,H,W)
      - mu: mean vector (B, bottleneck_dim)
      - logvar: log-variance vector (B, bottleneck_dim)
      - x_ref: reference input used for reconstruction loss (cropped/padded version)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.encoder = ConvNeXtUNetEncoder2D(
            in_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=cfg.skip_dropout_p,
            skip_alpha=cfg.skip_alpha,
        )

        self.decoder = ConvNeXtUNetDecoder2D(
            out_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_transpose_conv=cfg.use_transpose_conv,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=cfg.skip_dropout_p,
            skip_dropout_ps=cfg.skip_dropout_ps,
            skip_alpha=cfg.skip_alpha,
        )

        # Lazy FC layers (depend on latent spatial size)
        self.fc_mu: Optional[nn.Linear] = None
        self.fc_logvar: Optional[nn.Linear] = None
        self.fc_decode: Optional[nn.Linear] = None
        self._latent_hw: Optional[Tuple[int, int]] = None

    def _ensure_fcs(self, latent_hw: Tuple[int, int], device: torch.device):
        """Lazily create bottleneck fully-connected layers when latent size changes."""
        if self._latent_hw == latent_hw and self.fc_mu is not None:
            return

        self._latent_hw = latent_hw
        flat = int(self.cfg.z_channels * math.prod(latent_hw))

        self.fc_mu = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_logvar = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_decode = nn.Linear(self.cfg.bottleneck_dim, flat).to(device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder -> bottleneck -> decoder."""
        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(f"Expected C={self.cfg.in_channels}, got C={x.shape[1]}")

        x = x.float()
        device = x.device
        B = x.shape[0]
        ref_hw = tuple(x.shape[-2:])

        multiple = 2 ** self.cfg.n_levels
        x_pad, pad = self._pad_to_multiple(x, multiple)

        h, skips = self.encoder(x_pad)
        latent_hw = tuple(h.shape[-2:])
        self._ensure_fcs(latent_hw, device)

        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        z = self.reparameterize(mu, logvar)

        h_dec = self.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_hw)
        self.decoder.set_skips(skips)
        recon = self.decoder(h_dec)

        recon = self._crop_like(recon, ref_hw)
        x_ref = self._crop_like(x_pad, ref_hw) if sum(pad) else x

        return {"recon": recon, "mu": mu, "logvar": logvar, "x_ref": x_ref}

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
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1,C,H,W)
            single = True
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected (C,H,W) or (B,C,H,W), got {tuple(x.shape)}")

        if clamp_01:
            x = x.clamp(0.0, 1.0)

        x = x.to(device)
        #model.train()      # wichtig!

        with torch.no_grad():
            ref_hw = tuple(x.shape[-2:])
            multiple = 2 ** self.cfg.n_levels
            x_pad, pad = self._pad_to_multiple(x, multiple)

            h, skips = model.encoder(x_pad)
            latent_hw = tuple(h.shape[-2:])
            model._ensure_fcs(latent_hw, device)

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

            h_dec = model.fc_decode(z).reshape(B * n, self.cfg.z_channels, *latent_hw)


            alpha_skips = float(self.cfg.skip_alpha)  # 0.0=starke Variation, 0.2=leicht, 1.0=Rekonstruktion

            if alpha_skips <= 0:
                model.decoder.set_skips(None)
            else:
                # The decoder applies skip_alpha. Pass the raw encoder skips here
                # so posterior generation uses the same scaling as training.
                rep_skips = [sk.repeat_interleave(n, dim=0) for sk in skips]
                model.decoder.set_skips(rep_skips)

            
            recon = model.decoder(h_dec)
            recon = self._crop_like(recon, ref_hw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            recon = recon.view(B, n, self.cfg.in_channels, *ref_hw)

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
        """Warm up the model to initialize lazy FC layers.

        shape must be (C,H,W).
        """
        if not (isinstance(shape, (tuple, list)) and len(shape) == 3):
            raise ValueError(f"shape must be (C,H,W), got: {shape}")

        C, H, W = map(int, shape)
        if min(C, H, W) <= 0:
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
            x = torch.zeros((1, C, H, W), device=device, dtype=dtype)
            _ = self(x)

        if was_training:
            self.train()

        return self

    def _generate_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor, None] = None,
        *,
        out_hw: tuple[int, int] | None = None,
        variation_strength: float = 1.0,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Generate ONE synthetic sample via *prior sampling* (no input sample required).

        Samples:
            z ~ N(0, I)  (scaled by variation_strength), then decode to image space.

        Parameters:
        - out_hw: output (H, W). If None, tries cfg.sample_hw or cfg.image_hw, else defaults to (256,256).
        - variation_strength: prior diversity strength (1.0 is standard; <1.0 more conservative; >1.0 more diverse).
        - clamp_01: clamp outputs to [0,1].
        - return_torch: return torch.Tensor instead of np.ndarray.

        Output:
        - (C, H, W)
        """
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        
        # pick output size
        if out_hw is None and sample is not None:
            out_hw = tuple(self._extract_x(sample).shape[-2:])
        if not (isinstance(out_hw, (tuple, list)) and len(out_hw) == 2):
            raise ValueError(f"out_hw must be (H,W), got {out_hw}")
        H, W = int(out_hw[0]), int(out_hw[1])
        if H <= 0 or W <= 0:
            raise ValueError(f"out_hw must be positive, got {out_hw}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        # Ensure decoder doesn't expect encoder skips (we have none for pure prior sampling)
        model.decoder.set_skips(None)


        # Compute latent spatial size (assuming 2x downsample per level)
        down = 2 ** int(self.cfg.n_levels)

        # Pad to be divisible by down (so latent grid is integer)
        pad_h = (down - (H % down)) % down
        pad_w = (down - (W % down)) % down
        H_pad, W_pad = H + pad_h, W + pad_w
        latent_hw = (H_pad // down, W_pad // down)

        # Determine latent vector dim for fc_decode (matches your fc_mu/fc_logvar output)
        z_dim = int(self.cfg.bottleneck_dim)

        with torch.no_grad():
            model._ensure_fcs(latent_hw, device)

            # Prior sampling: z ~ N(0, I)
            if variation_strength == 0.0:
                z = torch.zeros((1, z_dim), device=device)
            else:
                z = torch.randn((1, z_dim), device=device) * float(variation_strength)

            # Map z -> decoder feature map and decode
            h_dec = model.fc_decode(z).reshape(1, int(self.cfg.z_channels), *latent_hw)
            recon = model.decoder(h_dec)  # (1, C, H_pad, W_pad) typically

            # Crop back to requested size and drop batch dim
            recon = recon[..., :H, :W].squeeze(0)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()

        if return_torch:
            return recon, target_mask_generator.create_target_mask(synth_anomaly_image=recon)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, target_mask_generator.create_target_mask(synth_anomaly_image=recon_np)
