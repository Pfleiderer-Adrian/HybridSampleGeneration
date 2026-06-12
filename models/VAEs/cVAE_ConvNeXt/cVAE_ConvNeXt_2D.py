from __future__ import annotations

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

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, Union, List
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models.model_interface import HybridModelInterface
from synthesizer.mask_manipulation import to_one_hot_2D


class SPADE2D(nn.Module):
    """Spatially-Adaptive Normalization (SPADE) for 2D data."""
    def __init__(self, norm_nc: int, label_nc: int, hidden_nc: int=128, kernel_size: int=3):
        super().__init__()
        # layer norm -> norm over all channels; affine=False: no params here as params should only come from SPADE
        self.no_param_instance_norm = nn.GroupNorm(num_groups=1, num_channels=norm_nc, affine=False)

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, kernel_size=kernel_size, padding=pw),
            nn.GELU()
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=pw)
    
    def forward(self, x: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.no_param_instance_norm(x)

        # scale mask to x's resolution, use 'nearest' if mask is one-hot encoded
        if tgt_mask.shape[-2:] != x.shape[-2:]:
            tgt_mask = F.interpolate(tgt_mask, size=x.shape[-2:], mode='nearest')

        activation = self.mlp_shared(tgt_mask)
        gamma = self.mlp_gamma(activation)
        beta = self.mlp_beta(activation)

        return normalized * (1 + gamma) + beta


class DropPath(nn.Module):
    """Stochastic Depth (per sample)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * random_tensor


def _best_gn_groups(default_groups: int, channels: int) -> int:
    """Choose a GroupNorm group count that divides channels."""
    g = min(default_groups, channels)
    while g > 1 and (channels % g != 0):
        g -= 1
    return g


class ConvNeXtBlock2D(nn.Module):
    """ConvNeXt-style block for 2D images (channels-first)."""

    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        gn_groups: int = 8,
        drop_path: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=True,
        )

        groups = _best_gn_groups(gn_groups, channels)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)

        hidden = int(channels * mlp_ratio)
        self.pwconv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return residual + x


class ConvNeXtSPADEBlock2D(nn.Module):
    """ConvNeXt-style block for 2D images (channels-first).
    SPADE instead of GroupNorm.
    """

    def __init__(
        self,
        channels: int,
        num_anomaly_classes: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=True,
        )

        self.spade = SPADE2D(norm_nc=channels, label_nc=num_anomaly_classes)

        hidden = int(channels * mlp_ratio)
        self.pwconv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.spade(x, mask) # use mask here
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return residual + x


def _upsample_block2d(in_ch: int, out_ch: int, scale: int, use_transpose_conv: bool, gn_groups: int) -> nn.Sequential:
    if use_transpose_conv:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale, stride=scale, padding=0, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, out_ch), num_channels=out_ch, eps=1e-6),
            nn.GELU(),
        )

    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, out_ch), num_channels=out_ch, eps=1e-6),
        nn.GELU(),
    )


class ConvNeXtUNetEncoder2D(nn.Module):
    """ConvNeXt2D encoder with U-Net skip outputs."""

    def __init__(
        self,
        in_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_multires_skips: bool = True,
        leak: float = 0.2,
        gn_groups: int = 8,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        skip_dropout_p: float = 0.0,
        skip_alpha: float = 1.0,
    ):
        super().__init__()
        self.n_levels = n_levels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, 8), num_channels=8, eps=1e-6),
            nn.GELU(),
        )

        self.blocks: nn.ModuleList = nn.ModuleList()
        self.downs: nn.ModuleList = nn.ModuleList()

        total_blocks = n_levels * n_res_blocks
        if drop_path_rate and drop_path_rate > 0 and total_blocks > 0:
            dp_rates = torch.linspace(0.0, float(drop_path_rate), steps=total_blocks).tolist()
        else:
            dp_rates = [0.0] * total_blocks
        dp_i = 0

        for i in range(n_levels):
            ch = 2 ** (i + 3)       # 8, 16, 32, 64...
            ch_next = 2 ** (i + 4)  # 16, 32, 64, 128...

            stage = []
            for _ in range(n_res_blocks):
                stage.append(ConvNeXtBlock2D(ch, mlp_ratio=4.0, gn_groups=gn_groups, drop_path=dp_rates[dp_i], dropout=dropout))
                dp_i += 1
            self.blocks.append(nn.Sequential(*stage))

            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch_next, kernel_size=2, stride=2, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch_next), num_channels=ch_next, eps=1e-6),
                    nn.GELU(),
                )
            )

        bottom_ch = 2 ** (n_levels + 3)
        self.to_z = nn.Conv2d(bottom_ch, z_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []

        for i in range(self.n_levels):
            x = self.blocks[i](x)
            skips.append(x)
            x = self.downs[i](x)

        h = self.to_z(x)
        return h, skips


class ConvNeXtSPADEUNetDecoder2D(nn.Module):
    """ConvNeXtSPADE2D decoder with U-Net skips."""
    
    def __init__(
        self,
        out_channels: int,
        n_res_blocks: int,
        n_spade_blocks: int,
        n_levels: int,
        z_channels: int,
        num_anomaly_classes: int,
        use_multires_skips: bool = True,
        leak: float = 0.2,
        use_transpose_conv: bool = True,
        gn_groups: int = 8,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        skip_dropout_p: float = 0.0,
        skip_alpha: float = 1.0,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_spade_blocks = n_spade_blocks
        self.use_transpose_conv = use_transpose_conv
        self._skips: Optional[List[torch.Tensor]] = None
        self.skip_dropout_p = float(skip_dropout_p)
        self.skip_alpha = float(skip_alpha)

        self.bottom_ch = 2 ** (n_levels + 3)
        self.from_z = nn.Sequential(
            nn.Conv2d(z_channels, self.bottom_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, self.bottom_ch), num_channels=self.bottom_ch, eps=1e-6),
            nn.GELU(),
        )

        self.ups: nn.ModuleList = nn.ModuleList()
        self.fuse: nn.ModuleList = nn.ModuleList()
        self.blocks: nn.ModuleList = nn.ModuleList()

        total_blocks = n_levels * n_res_blocks
        if drop_path_rate and drop_path_rate > 0 and total_blocks > 0:
            dp_rates = torch.linspace(0.0, float(drop_path_rate), steps=total_blocks).tolist()
        else:
            dp_rates = [0.0] * total_blocks
        dp_i = 0

        prev_ch = self.bottom_ch
        for i in range(n_levels):
            ch = 2 ** (n_levels - i + 2)

            self.ups.append(_upsample_block2d(prev_ch, ch, scale=2, use_transpose_conv=use_transpose_conv, gn_groups=gn_groups))

            skip_ch = 2 ** (n_levels - i + 2)
            self.fuse.append(
                nn.Sequential(
                    nn.Conv2d(ch + skip_ch, ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch), num_channels=ch, eps=1e-6),
                    nn.GELU(),
                )
            )

            # use SPADE here
            num_spade = min(self.n_spade_blocks, n_res_blocks)
            stage = nn.ModuleList()
            for j in range(n_res_blocks):
                if j < num_spade:
                    stage.append(ConvNeXtSPADEBlock2D(
                        channels=ch, num_anomaly_classes=num_anomaly_classes, 
                        mlp_ratio=4.0, drop_path=dp_rates[dp_i], dropout=dropout))
                else:
                    stage.append(ConvNeXtBlock2D(
                        channels=ch, mlp_ratio=4.0, gn_groups=gn_groups, 
                        drop_path=dp_rates[dp_i], dropout=dropout))
                dp_i += 1
            self.blocks.append(stage)

            prev_ch = ch

        self.out = nn.Conv2d(prev_ch, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def set_skips(self, skips: Optional[List[torch.Tensor]]) -> None:
        self._skips = skips

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode latent feature map `z` into an image using `mask` for SPADE."""
        x = self.from_z(z)

        skips = self._skips
        if skips is not None and len(skips) != self.n_levels:
            raise ValueError(f"Expected {self.n_levels} skips, got {len(skips)}")

        for i in range(self.n_levels):
            x = self.ups[i](x)

            if skips is None:
                # No skips provided -> treat as zeros
                skip_ch = 2 ** (self.n_levels - i + 2)
                skip = torch.zeros(
                    (x.shape[0], skip_ch, x.shape[-2], x.shape[-1]),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                skip = skips[-1 - i]

            # Align spatial sizes
            if x.shape[-2:] != skip.shape[-2:]:
                target = (min(x.shape[-2], skip.shape[-2]), min(x.shape[-1], skip.shape[-1]))
                x = HybridModelInterface._crop_like(x, target)
                skip = HybridModelInterface._crop_like(skip, target)

            # Apply skip scaling
            if self.skip_alpha != 1.0:
                skip = skip * self.skip_alpha

            # Skip-Dropout manually implemented
            p = self.skip_dropout_p
            if p > 0.0 and self.training:
                keep_prob = 1.0 - p
                drop_mask = (torch.rand((skip.shape[0], 1, 1, 1), device=skip.device, dtype=skip.dtype) < keep_prob).to(skip.dtype)
                skip = skip * drop_mask / max(keep_prob, 1e-6)

            x = torch.cat([x, skip], dim=1)
            x = self.fuse[i](x)

            # mask in every SPADE block
            num_spade = min(self.n_spade_blocks, len(self.blocks[i]))
            for j, block in enumerate(self.blocks[i]):
                if j < num_spade:
                    x = block(x, mask)  # SPADE block
                else:
                    x = block(x)    # normal block

        return self.out(x)


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
    use_multires_skips: bool = True
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
    skip_alpha: float = 1.0      # Scale skips (0.0 disables skips, 0.2 keeps small guidance)


class ConvNeXtcVAE2D(HybridModelInterface):
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
            use_multires_skips=cfg.use_multires_skips,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=getattr(cfg, 'skip_dropout_p', 0.0),
            skip_alpha=getattr(cfg, 'skip_alpha', 1.0),
        )

        self.decoder = ConvNeXtSPADEUNetDecoder2D(
            out_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_spade_blocks=cfg.n_spade_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            num_anomaly_classes=cfg.num_anomaly_classes,
            use_multires_skips=cfg.use_multires_skips,
            use_transpose_conv=cfg.use_transpose_conv,
            drop_path_rate=cfg.drop_path_rate,
            dropout=cfg.dropout,
            skip_dropout_p=getattr(cfg, 'skip_dropout_p', 0.0),
            skip_alpha=getattr(cfg, 'skip_alpha', 1.0),
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

    def _extract_inputs(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract x, ori_mask, and tgt_mask from batch."""
        if isinstance(batch, dict):
            x = batch.get("img", batch.get("x"))
            ori_mask = batch.get("ori_mask", batch.get("mask"))
            tgt_mask = batch.get("tgt_mask")
            if x is not None and ori_mask is not None and tgt_mask is not None:
                return torch.as_tensor(x), torch.as_tensor(ori_mask), torch.as_tensor(tgt_mask)
            else:
                raise ValueError(
                f"Dataloader returned a dict, but expected keys are missing. "
                f"Found keys: {list(batch.keys())}. "
                f"Expected combinations of ('img' or 'x') and ('mask' or 'label') and 'tgt_mask'."
            )

        if isinstance(batch, (tuple, list)) and len(batch) >= 3:
            return torch.as_tensor(batch[0]), torch.as_tensor(batch[1]), torch.as_tensor(batch[2])

        raise TypeError(f"Unknown batch type: {type(batch)}")

    def fit_epoch(
        self,
        train_dataloader,
        val_dataloader,
        optimizer,
        *,
        epoch_idx: Optional[int] = None,
        log_every=1,
        grad_clip_norm: Optional[float] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> tuple[dict, dict]:
        device = torch.device(device)
        model = self.to(device)

        def run_epoch(loader: Iterable, training: bool) -> Dict[str, float]:
            model.train(training)
            run = {"total": 0.0, "recon": 0.0, "kl": 0.0, "recon_weighted": 0.0, "kl_weighted": 0.0, "kl_raw": 0.0}
            n = 0

            pbar = tqdm(loader, desc=("train" if training else "val"), leave=False, dynamic_ncols=True)
            for step, batch in enumerate(pbar, start=1):
                x, ori_mask, _ = self._extract_inputs(batch=batch)
                x = x.to(device, non_blocking=True)
                ori_mask = ori_mask.to(device, non_blocking=True)

                if x.ndim != 4 or ori_mask.ndim not in [3, 4]:
                    raise ValueError(f"Expected (B,C,H,W) from dataloader, got {tuple(x.shape)} and mask {tuple(ori_mask.shape)}.")

                if training:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    out = model(x, ori_mask, ori_mask)
                    losses = model.loss(out)
                    loss = losses["total"]

                    if training:
                        loss.backward()
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()

                for k in run:
                    if k in losses:
                        run[k] += float(losses[k].detach().item())
                n += 1

                if log_every and (step % log_every == 0):
                    postfix = {"total": f"{float(losses['total'].detach()):.6f}"}
                    postfix["recon"] = f"{float(losses['recon'].detach()):.6f}"
                    postfix["kl"] = f"{float(losses['kl'].detach()):.6f}"
                    pbar.set_postfix(postfix)

            denom = max(1, n)
            return {k: run[k] / denom for k in run}

        tr = run_epoch(train_dataloader, training=True)
        va = {}
        if val_dataloader is not None:
            with torch.no_grad():
                va = run_epoch(val_dataloader, training=False)

        return tr, va

    def generate_synth_sample(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        original_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        *,
        n: int = 1,
        s: float = 0.5,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")

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
                target_mask = sample.get("tgt_mask", original_mask)
        else:
            x = torch.as_tensor(sample).float()

        if original_mask is None:
            raise ValueError("original_mask is required for conditional generation.")
        
        ori_mask = torch.as_tensor(original_mask)
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

            if s == 0.0:
                z = mu.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
            else:
                eps = torch.randn((B, n, mu.shape[-1]), device=device, dtype=mu.dtype)
                z = (mu.unsqueeze(1) + (s * std).unsqueeze(1) * eps).reshape(B * n, -1)

            h_dec = model.fc_decode(z).reshape(B * n, self.cfg.z_channels, *latent_hw)

            alpha_skips = float(getattr(self.cfg, 'skip_alpha', 0.2))
            if alpha_skips <= 0:
                model.decoder.set_skips(None)
            else:
                rep_skips = [(alpha_skips * sk).repeat_interleave(n, dim=0) for sk in skips]
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

    def warmup(self, shape, device=None, dtype=None):
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

    def generate_synth_sample_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        *,
        s: float = 1.0,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        if isinstance(sample, dict):
            target_mask = sample.get("tgt_mask", sample.get("ori_mask", sample.get("mask")))
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
            z_dim = int(getattr(self.cfg, "bottleneck_dim", 256))

            if s == 0.0:
                z = torch.zeros((B, z_dim), device=device)
            else:
                z = torch.randn((B, z_dim), device=device) * float(s)

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

if __name__ == "__main__":
    # Quick sanity check
    cfg = Config(in_channels=1, num_anomaly_classes=4, n_res_blocks=2, n_levels=4, z_channels=64, bottleneck_dim=64)
    model = ConvNeXtcVAE2D(cfg=cfg)
    
    x = torch.randn(2, 1, 128, 128)
    mask = torch.zeros(2, 1, 128, 128, dtype=torch.long)
    
    out = model(x, mask)
    print("Forward Pass output:")
    print({k: tuple(v.shape) for k, v in out.items()})

    # Posterior sampling: generate 3 variants per item
    variants = model.generate_synth_sample(x[0], original_mask=mask[0], n=3, s=0.3, return_torch=True)
    print("Variants (posterior sampling) shape:", tuple(variants.shape))
    
    # Prior sampling
    prior = model.generate_synth_sample_prior(target_mask=mask[0], s=1.0, return_torch=True)
    print("Prior sampling shape:", tuple(prior.shape))
