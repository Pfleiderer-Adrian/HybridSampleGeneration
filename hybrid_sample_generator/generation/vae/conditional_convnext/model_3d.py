"""ConvNeXt3D-U-Net conditional VAE

Features:
- mask (one-hot encoded) concatenated to corresponding input
- ConvNeXt3D blocks (depthwise conv + pointwise MLP)
- BatchNorm -> GroupNorm (more stable for 3D and small batch sizes)
- True U-Net skip connections (feature concatenation)
- SPADE blocks for mask integration in decoder

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, Union, List
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_sample_generator.generation.vae.base import HybridVAEBase
from hybrid_sample_generator.imaging.masks.encoding import to_one_hot_3D
from hybrid_sample_generator.imaging.masks.transforms import TransformGenerator


from hybrid_sample_generator.generation.vae.convnext.layers_3d import (
    ConvNeXtBlock3D,
    ConvNeXtUNetEncoder3D,
    DropPath,
    _best_gn_groups,
    _upsample_block3d,
)
from .spade_3d import SPADE3D, ConvNeXtSPADEBlock3D, ConvNeXtSPADEUNetDecoder3D

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
    num_anomaly_classes: int = None
    n_res_blocks: int = 8
    n_spade_blocks: int = 2 # how many of the res blocks should use spade? for every upscale?
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

class ConvNeXtcVAE3D(HybridVAEBase):
    """3D ConvNeXt-U-Net VAE with SPADE."""

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.num_anomaly_classes is None:
            raise ValueError("Config.num_anomaly_classes must be set for ConvNeXtcVAE3D.")
        self.cfg = cfg
        self.in_channels = cfg.in_channels

        # encoder gets real mask as additional input (concatenated)
        enc_in_channels = cfg.in_channels + cfg.num_anomaly_classes   

        # Encoder returns (h, skips)
        self.encoder = ConvNeXtUNetEncoder3D(
            in_channels=enc_in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
        )

        # Decoder reconstructs from latent feature map; skips are set each forward
        self.decoder = ConvNeXtSPADEUNetDecoder3D(
            out_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            n_spade_blocks=cfg.n_spade_blocks,
            z_channels=cfg.z_channels,
            num_anomaly_classes=cfg.num_anomaly_classes,
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

    def forward(self, x: torch.Tensor, ori_mask: torch.Tensor, tgt_mask: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder (x and ori_mask) -> bottleneck -> decoder with SPADE using tgt_mask."""
        if tgt_mask is None:
            tgt_mask = ori_mask
        ori_mask = to_one_hot_3D(ori_mask, self.cfg.num_anomaly_classes)
        tgt_mask = to_one_hot_3D(tgt_mask, self.cfg.num_anomaly_classes)
        if x.ndim != 5 or ori_mask.ndim != 5 or tgt_mask.ndim != 5:
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
        if sum(pad) > 0:
            ori_mask_pad = F.pad(ori_mask, pad, mode="constant", value=0.0)
            tgt_mask_pad = F.pad(tgt_mask, pad, mode="constant", value=0.0)
        else:
            ori_mask_pad = ori_mask
            tgt_mask_pad = tgt_mask

        # Encode -> (latent feature map, skips)
        enc_in = torch.cat([x_pad, ori_mask_pad], dim=1)    # concat x and ori_mask for encoder input
        h, skips = self.encoder(enc_in)
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
        
        recon = self.decoder(h_dec, tgt_mask_pad)

        # Crop recon back to original spatial size
        recon = self._crop_like(recon, ref_dhw)
        x_ref = self._crop_like(x_pad, ref_dhw) if sum(pad) else x

        return {"recon": recon, "mu": mu, "logvar": logvar, "x_ref": x_ref}

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
        target_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,  # use target mask if you want spatial variation
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
            Sample dict containing "img", "ori_mask" and optionally "tgt_mask",
            or raw (C,D,H,W) / (B,C,D,H,W).
        original_mask, target_mask:
            Required only when sample is a raw tensor/array.

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
        
        # --- WICHTIG: KEIN .float() mehr für die Masken! ---
        ori_mask = torch.as_tensor(original_mask)
        if target_mask is None and target_mask_generator is not None:
            target_mask = target_mask_generator.create_target_mask(original_mask=original_mask, conditional=True)

        if target_mask is None:
            tgt_mask = ori_mask
        else:
            tgt_mask = torch.as_tensor(target_mask)
        tgt_mask_return = tgt_mask

        single = False

        if x.ndim == 4:
            x = x.unsqueeze(0)  # (1,C,D,H,W)
            single = True
        elif x.ndim != 5:
            raise ValueError(f"Expected (C,D,H,W) or (B,C,D,H,W), got {tuple(x.shape)}")

        if clamp_01:
            x = x.clamp(0.0, 1.0)

        x = x.to(device)
        
        # to device and then one-hot
        ori_mask = to_one_hot_3D(ori_mask.to(device), self.cfg.num_anomaly_classes)
        tgt_mask = to_one_hot_3D(tgt_mask.to(device), self.cfg.num_anomaly_classes)

        with torch.no_grad():
            # --- same preprocessing as forward() ---
            ref_dhw = tuple(x.shape[-3:])
            multiple = 2 ** self.cfg.n_levels
            x_pad, pad = self._pad_to_multiple(x, multiple)
            if sum(pad) > 0:
                ori_mask_pad = F.pad(ori_mask, pad, mode="constant", value=0.0)
                tgt_mask_pad = F.pad(tgt_mask, pad, mode="constant", value=0.0)
            else:
                ori_mask_pad = ori_mask
                tgt_mask_pad = tgt_mask

            enc_in = torch.cat([x_pad, ori_mask_pad], dim=1)
            # Encode once
            h, skips = model.encoder(enc_in)
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

            # need tgt_mask n times for for n reconstructions
            tgt_mask_pad_rep = tgt_mask_pad.repeat_interleave(n, dim=0)
            recon = model.decoder(h_dec, tgt_mask_pad_rep)  # decoder gets tgt_mask
            recon = self._crop_like(recon, ref_dhw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            # Reshape back to (B,n,C,D,H,W)
            recon = recon.view(B, n, self.in_channels, *ref_dhw)

            if single:
                recon = recon.squeeze(0)  # (n,C,D,H,W)
                recon = recon.squeeze(0)

        if return_torch:
            return recon, tgt_mask_return.to(recon.device)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        tgt_mask_np = tgt_mask_return.cpu().numpy().astype(np.uint8, copy=False)
        return recon_np, tgt_mask_np

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
            mask = torch.zeros((1, D, H, W), device=device, dtype=torch.long)
            _ = self(x, mask)

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
        """
        Generate synthetic 3D samples via *prior sampling* conditioned on a target mask.
        (No input image required).

        Samples:
            z ~ N(0, I) (scaled by variation_strength), then decode to 3D volume space conditioned on target_mask.

        Parameters:
        - sample: sample dict containing "tgt_mask", or a label mask with
          shape (D,H,W), (C,D,H,W) or (B,C,D,H,W). Defines the classes and output spatial size.
        - variation_strength: prior diversity strength (1.0 is standard; <1.0 more conservative).
        - clamp_01: clamp outputs to [0,1].
        - return_torch: return torch.Tensor instead of np.ndarray.

        Output:
        - (C, D, H, W) if input was 4D
        - (B, C, D, H, W) if input was 5D
        """
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        model.decoder.set_skips(None)

        if isinstance(sample, dict):
            target_mask = sample.get("tgt_mask")
            original_mask = sample.get("ori_mask", sample.get("mask"))
            if target_mask is None and target_mask_generator is not None:
                target_mask = target_mask_generator.create_target_mask(original_mask=original_mask, conditional=True)
            if target_mask is None:
                target_mask = original_mask
            if target_mask is None:
                raise KeyError("Conditional prior sample dict must contain 'tgt_mask' or 'ori_mask'.")
        else:
            target_mask = sample

        tgt_mask = torch.as_tensor(target_mask)
        tgt_mask_return = tgt_mask
        single = False

        if tgt_mask.ndim in [3, 4]:
            # if 3D (D,H,W) or 4D (1,D,H,W)/(C,D,H,W) -> make it batched
            if tgt_mask.ndim == 3:
                tgt_mask = tgt_mask.unsqueeze(0)
            tgt_mask = tgt_mask.unsqueeze(0)
            single = True
        elif tgt_mask.ndim == 5:
            pass
        else:
            raise ValueError(f"Expected target_mask (D,H,W), (C,D,H,W) or (B,C,D,H,W), got {tuple(tgt_mask.shape)}")

        tgt_mask = tgt_mask.to(device)
        tgt_mask_oh = to_one_hot_3D(tgt_mask, self.cfg.num_anomaly_classes)

        with torch.no_grad():
            ref_dhw = tuple(tgt_mask_oh.shape[-3:])
            multiple = 2 ** self.cfg.n_levels
            
            # Pad the target mask so spatial dims are divisible by the downsampling factor
            tgt_mask_pad, pad = self._pad_to_multiple(tgt_mask_oh, multiple)

            # Calculate latent dimensions based on the padded mask
            latent_dhw = (
                tgt_mask_pad.shape[2] // multiple,
                tgt_mask_pad.shape[3] // multiple,
                tgt_mask_pad.shape[4] // multiple
            )
            
            model._ensure_fcs(latent_dhw, device)

            B = tgt_mask_oh.shape[0]
            z_dim = int(self.cfg.bottleneck_dim)

            # Prior sampling: z ~ N(0, I)
            if variation_strength == 0.0:
                z = torch.zeros((B, z_dim), device=device)
            else:
                z = torch.randn((B, z_dim), device=device) * float(variation_strength)

            # Map z -> decoder feature map
            h_dec = model.fc_decode(z).reshape(B, int(self.cfg.z_channels), *latent_dhw)

            # Decode conditioned on the target mask
            recon = model.decoder(h_dec, tgt_mask_pad)
            
            # Crop back to exact requested size
            recon = self._crop_like(recon, ref_dhw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            # Squeeze batch dimension if input lacked it
            if single:
                recon = recon.squeeze(0)

        if return_torch:
            return recon, tgt_mask_return.to(recon.device)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        tgt_mask_np = tgt_mask_return.cpu().numpy().astype(np.uint8, copy=False)
        return recon_np, tgt_mask_np
    
