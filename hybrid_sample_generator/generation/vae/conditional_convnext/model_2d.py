"""ConvNeXt2D-U-Net conditional VAE

Features:
- mask (one-hot encoded) concatenated to corresponding input
- ConvNeXt2D blocks (depthwise conv + pointwise MLP)
- GroupNorm instead of BatchNorm (stable for small batch sizes)
- True U-Net skip connections (feature concatenation)
- SPADE blocks for mask integration in decoder
- Posterior sampling for *lightly varied* variants around a given input

Shapes:
- Input x: (B, C, H, W)
- Input mask: (B, 1, H, W) or (B, H, W)
- Output recon: (B, C, H, W)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union, List
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from hybrid_sample_generator.generation.vae.base import HybridVAEBase
from hybrid_sample_generator.imaging.masks.encoding import to_one_hot_2D
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


from hybrid_sample_generator.generation.vae.convnext.layers_2d import ConvNeXtUNetEncoder2D
from .spade_2d import ConvNeXtSPADEUNetDecoder2D

# -------------------------
# VAE 2D conditional
# -------------------------

@dataclass
class Config:
    """Hyperparameters for ConvNeXtcVAE2D."""
    in_channels: int = None
    num_anomaly_classes: int = None
    n_res_blocks: int = 8
    n_spade_blocks: int = 2 # how many of the res blocks should use spade per upscale level
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


class ConvNeXtcVAE2D(HybridVAEBase):
    """2D ConvNeXt-U-Net VAE with SPADE for conditional generation."""

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.num_anomaly_classes is None:
            raise ValueError("Config.num_anomaly_classes must be set for ConvNeXtcVAE2D.")
        self.cfg = cfg
        self.in_channels = cfg.in_channels

        # encoder gets real mask as additional input (concatenated)
        enc_in_channels = cfg.in_channels + cfg.num_anomaly_classes

        self.encoder = ConvNeXtUNetEncoder2D(
            in_channels=enc_in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=cfg.skip_dropout_p,
            skip_alpha=cfg.skip_alpha,
        )

        self.decoder = ConvNeXtSPADEUNetDecoder2D(
            out_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_spade_blocks=cfg.n_spade_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            num_anomaly_classes=cfg.num_anomaly_classes,
            use_transpose_conv=cfg.use_transpose_conv,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=cfg.skip_dropout_p,
            skip_dropout_ps=cfg.skip_dropout_ps,
            skip_alpha=cfg.skip_alpha,
        )

        self.fc_mu: Optional[nn.Linear] = None
        self.fc_logvar: Optional[nn.Linear] = None
        self.fc_decode: Optional[nn.Linear] = None
        self._latent_hw: Optional[Tuple[int, int]] = None

    def _ensure_fcs(self, latent_hw: Tuple[int, int], device: torch.device):
        if self._latent_hw == latent_hw and self.fc_mu is not None:
            return

        self._latent_hw = latent_hw
        flat = int(self.cfg.z_channels * math.prod(latent_hw))

        self.fc_mu = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_logvar = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_decode = nn.Linear(self.cfg.bottleneck_dim, flat).to(device)

    def forward(self, x: torch.Tensor, ori_mask: torch.Tensor, tgt_mask: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        if tgt_mask is None:
            tgt_mask = ori_mask
            
        ori_mask = to_one_hot_2D(ori_mask, self.cfg.num_anomaly_classes)
        tgt_mask = to_one_hot_2D(tgt_mask, self.cfg.num_anomaly_classes)
        
        if x.ndim != 4 or ori_mask.ndim != 4 or tgt_mask.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(f"Expected C={self.cfg.in_channels}, got C={x.shape[1]}")

        x = x.float()
        device = x.device
        B = x.shape[0]
        ref_hw = tuple(x.shape[-2:])

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
        latent_hw = tuple(h.shape[-2:])

        self._ensure_fcs(latent_hw, device)

        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        z = self.reparameterize(mu, logvar)

        # Decode
        h_dec = self.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_hw)
        self.decoder.set_skips(skips)
        
        recon = self.decoder(h_dec, tgt_mask_pad)

        recon = self._crop_like(recon, ref_hw)
        x_ref = self._crop_like(x_pad, ref_hw) if sum(pad) else x

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
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1,C,H,W)
            single = True
        elif x.ndim != 4:
            raise ValueError(f"Expected (C,H,W) or (B,C,H,W), got {tuple(x.shape)}")

        if clamp_01:
            x = x.clamp(0.0, 1.0)

        x = x.to(device)
        
        ori_mask = to_one_hot_2D(ori_mask.to(device), self.cfg.num_anomaly_classes)
        tgt_mask = to_one_hot_2D(tgt_mask.to(device), self.cfg.num_anomaly_classes)

        with torch.no_grad():
            ref_hw = tuple(x.shape[-2:])
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

            alpha_skips = float(self.cfg.skip_alpha)
            if alpha_skips <= 0:
                model.decoder.set_skips(None)
            else:
                # The decoder applies skip_alpha. Pass the raw encoder skips here
                # so posterior generation uses the same scaling as training.
                rep_skips = [sk.repeat_interleave(n, dim=0) for sk in skips]
                model.decoder.set_skips(rep_skips)

            tgt_mask_pad_rep = tgt_mask_pad.repeat_interleave(n, dim=0)
            recon = model.decoder(h_dec, tgt_mask_pad_rep)
            recon = self._crop_like(recon, ref_hw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            recon = recon.view(B, n, self.cfg.in_channels, *ref_hw)

            if single:
                recon = recon.squeeze(0)
                recon = recon.squeeze(0)

        if return_torch:
            return recon, tgt_mask_return.to(recon.device)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        tgt_mask_np = tgt_mask_return.cpu().numpy().astype(np.uint8, copy=False)
        return recon_np, tgt_mask_np

    def warmup(self, shape, device=None, dtype=None, config=None):
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
            mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.long)
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
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        if isinstance(sample, dict):
            target_mask = sample.get("tgt_mask")
            original_mask = sample.get("ori_mask", sample.get("mask"))
            if target_mask is None and target_mask_generator is not None:
                target_mask = target_mask_generator.create_target_mask(original_mask=original_mask, conditional=True)
            if target_mask is None:
                target_mask = original_mask
            if target_mask is None:
                raise KeyError("Conditional prior sample dict must contain 'tgt_mask' or 'ori_mask'.")

        tgt_mask = torch.as_tensor(target_mask)
        tgt_mask_return = tgt_mask
        single = False

        if tgt_mask.ndim in [2, 3]:
            # if 2D (H,W) or 3D (1, H, W) -> make it batched
            if tgt_mask.ndim == 2:
                tgt_mask = tgt_mask.unsqueeze(0)
            tgt_mask = tgt_mask.unsqueeze(0)
            single = True
        elif tgt_mask.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected target_mask (H,W), (C,H,W) or (B,C,H,W), got {tuple(tgt_mask.shape)}")

        tgt_mask = tgt_mask.to(device)
        tgt_mask_oh = to_one_hot_2D(tgt_mask, self.cfg.num_anomaly_classes)

        model.decoder.set_skips(None)

        with torch.no_grad():
            ref_hw = tuple(tgt_mask_oh.shape[-2:])
            multiple = 2 ** self.cfg.n_levels
            
            tgt_mask_pad, pad = self._pad_to_multiple(tgt_mask_oh, multiple)

            latent_hw = (
                tgt_mask_pad.shape[2] // multiple,
                tgt_mask_pad.shape[3] // multiple
            )
            
            model._ensure_fcs(latent_hw, device)

            B = tgt_mask_oh.shape[0]
            z_dim = int(self.cfg.bottleneck_dim)

            if variation_strength == 0.0:
                z = torch.zeros((B, z_dim), device=device)
            else:
                z = torch.randn((B, z_dim), device=device) * float(variation_strength)

            h_dec = model.fc_decode(z).reshape(B, int(self.cfg.z_channels), *latent_hw)

            recon = model.decoder(h_dec, tgt_mask_pad)
            recon = self._crop_like(recon, ref_hw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            if single:
                recon = recon.squeeze(0)

        if return_torch:
            return recon, tgt_mask_return.to(recon.device)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        tgt_mask_np = tgt_mask_return.cpu().numpy().astype(np.uint8, copy=False)
        return recon_np, tgt_mask_np
