from __future__ import annotations

"""ConvNeXt3D-U-Net conditional VAE

Features:
- mask (one-hot encoded) concatenated to corresponding input
- ConvNeXt3D blocks (depthwise conv + pointwise MLP)
- BatchNorm -> GroupNorm (more stable for 3D and small batch sizes)
- True U-Net skip connections (feature concatenation)
- SPADE blocks for mask integration in decoder

"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, Union, List
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from synthesizer.mask_augmentation import to_one_hot_3D


# -------------------------
# helpers: padding/cropping
# -------------------------

def _compute_symmetric_pad(size: int, multiple: int) -> Tuple[int, int]:
    """Compute symmetric (left,right) padding so that `size` becomes divisible by `multiple`."""
    if multiple <= 1:
        return (0, 0)
    r = size % multiple
    if r == 0:
        return (0, 0)
    need = multiple - r
    left = need // 2
    right = need - left
    return left, right


def _pad_to_multiple_3d(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Pad (B,C,D,H,W) so D/H/W are divisible by `multiple` (symmetric, constant=0)."""
    d, h, w = x.shape[-3:]

    dL, dR = _compute_symmetric_pad(d, multiple)
    hL, hR = _compute_symmetric_pad(h, multiple)
    wL, wR = _compute_symmetric_pad(w, multiple)

    pad = (wL, wR, hL, hR, dL, dR)
    if sum(pad) == 0:
        return x, pad

    return F.pad(x, pad, mode="constant", value=0.0), pad


def _crop_like_3d(x: torch.Tensor, ref_dhw: Tuple[int, int, int]) -> torch.Tensor:
    """Center-crop last 3 dims of x to ref_dhw."""
    d_ref, h_ref, w_ref = ref_dhw
    d, h, w = x.shape[-3:]

    def sl(cur: int, ref: int):
        if cur == ref:
            return slice(None)
        start = (cur - ref) // 2
        return slice(start, start + ref)

    return x[..., sl(d, d_ref), sl(h, h_ref), sl(w, w_ref)]


# -------------------------
# blocks (3D)
# -------------------------

class SPADE3D(nn.Module):
    """Spatially-Adaptive Normalization (SPADE) for 3D data."""
    def __init__(self, norm_nc: int, label_nc: int, hidden_nc: int=128, kernel_size: int=3):
        super().__init__()
        # layer norm -> norm over all channels; affine=False: no params here as params should only come from SPADE
        # must be for one-hot encoding (channels)
        self.no_param_instance_norm = nn.GroupNorm(num_groups=1, num_channels=norm_nc, affine=False)

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, hidden_nc, kernel_size=kernel_size, padding=pw), # same padding
            nn.GELU()
        )
        # gamma and beta are now tensors
        self.mlp_gamma = nn.Conv3d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv3d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=pw)
    
    def forward(self, x: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.no_param_instance_norm(x)

        # scale mask to x's resolution, use 'nearest' if mask is one-hot encoded
        if tgt_mask.shape[-3:] != x.shape[-3:]:
            tgt_mask = F.interpolate(tgt_mask, size=x.shape[-3:], mode='nearest')

        activation = self.mlp_shared(tgt_mask)
        gamma = self.mlp_gamma(activation)
        beta = self.mlp_beta(activation)

        return normalized * (1 + gamma) + beta

# kopiert aus convnext3d
class ConvNeXtBlock3D(nn.Module):
    """ConvNeXt-style block for 3D volumes (channels-first).

    Design (simplified):
      depthwise conv (k=7) -> GroupNorm -> pointwise expand -> GELU -> pointwise project
      + residual

    Notes:
      - GroupNorm is used instead of LayerNorm to avoid channel-last permutations.
      - You can scale capacity via mlp_ratio.
    """

    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        gn_groups: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # Depthwise conv
        # NOTE: In 2D ConvNeXt commonly uses k=7. In 3D, k=7 means 7x7x7=343
        # kernel elements and can be prohibitively slow/heavy. For volumetric VAEs
        # (especially in CPU-bound inference/debug), a smaller kernel is a much
        # better default.
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        # GroupNorm (stable for small batch sizes)
        groups = min(gn_groups, channels)
        # Ensure divisibility
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)

        hidden = int(channels * mlp_ratio)
        self.pwconv1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

        # Optional stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return residual + x

class ConvNeXtSPADEBlock3D(nn.Module):
    """ConvNeXt-style block for 3D volumes (channels-first).
    SPADE instead of GroupNorm.
    """

    def __init__(
        self,
        channels: int,
        num_anomaly_classes: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        self.spade = SPADE3D(norm_nc=channels, label_nc=num_anomaly_classes)

        hidden = int(channels * mlp_ratio)
        self.pwconv1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

        # Optional stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.spade(x, mask) # use mask here
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return residual + x


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

class ConvNeXtUNetEncoder3D(nn.Module):
    """ConvNeXt3D encoder with U-Net skip outputs.

    forward(x) returns:
      - h: deepest latent feature map (B, z_channels, d', h', w')
      - skips: list of feature maps at each resolution (for decoder), length n_levels

      base=8, then 16, 32, 64, ...
    """

    def __init__(
        self,
        in_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_multires_skips: bool = True,  # kept for API compatibility (not used)
        leak: float = 0.2,  # kept for API compatibility
        gn_groups: int = 8,
    ):
        super().__init__()
        self.n_levels = n_levels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=min(gn_groups, 8), num_channels=8, eps=1e-6),
            nn.GELU(),
        )

        self.blocks: nn.ModuleList = nn.ModuleList()
        self.downs: nn.ModuleList = nn.ModuleList()

        for i in range(n_levels):
            ch = 2 ** (i + 3)      # 8,16,32,64...
            ch_next = 2 ** (i + 4)  # 16,32,64,128...

            stage = []
            for _ in range(n_res_blocks):
                stage.append(ConvNeXtBlock3D(ch, mlp_ratio=4.0, gn_groups=gn_groups))
            self.blocks.append(nn.Sequential(*stage))

            # Downsample
            self.downs.append(
                nn.Sequential(
                    nn.Conv3d(ch, ch_next, kernel_size=2, stride=2, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch_next), num_channels=ch_next, eps=1e-6),
                    nn.GELU(),
                )
            )

        # Bottom projection to z_channels
        bottom_ch = 2 ** (n_levels + 3)
        self.to_z = nn.Conv3d(bottom_ch, z_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []

        for i in range(self.n_levels):
            x = self.blocks[i](x)
            skips.append(x)
            x = self.downs[i](x)

        h = self.to_z(x)
        return h, skips

class ConvNeXtSPADEUNetDecoder3D(nn.Module):
    """ConvNeXtSPADE3D decoder with U-Net skips.
    """
    def __init__(
        self,
        out_channels: int,
        n_res_blocks: int,
        n_spade_blocks: int,
        n_levels: int,
        z_channels: int,
        num_anomaly_classes: int,
        use_multires_skips: bool = True,  # kept for API compatibility (not used)
        leak: float = 0.2,  # kept for API compatibility
        use_transpose_conv: bool = True,
        skip_dropout_p: float = 0.0,
        skip_alpha: float = 1.0,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.use_transpose_conv = use_transpose_conv
        self.skip_dropout_p = float(skip_dropout_p)
        self.skip_alpha = float(skip_alpha)
        self._skips: Optional[List[torch.Tensor]] = None
        self.n_spade_blocks = n_spade_blocks

        # Project latent channels up to bottom channels
        self.bottom_ch = 2 ** (n_levels + 3)
        self.from_z = nn.Sequential(
            nn.Conv3d(z_channels, self.bottom_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, self.bottom_ch), num_channels=self.bottom_ch, eps=1e-6),
            nn.GELU(),
        )

        self.ups: nn.ModuleList = nn.ModuleList()
        self.fuse: nn.ModuleList = nn.ModuleList()
        self.blocks: nn.ModuleList = nn.ModuleList()

        prev_ch = self.bottom_ch
        for i in range(n_levels):
            # At decoder step i we go to channel count of encoder level (reversed).
            # This matches the original schedule:
            #   n_filters = 2**(n_levels - i + 2)
            ch = 2 ** (n_levels - i + 2)

            self.ups.append(_upsample_block3d(prev_ch, ch, scale=2, use_transpose_conv=use_transpose_conv, gn_groups=gn_groups))

            # Skip concat: (ch + skip_ch) -> ch
            # skip_ch corresponds to encoder level (n_levels-1-i) channels: 2**((n_levels-1-i)+3) = 2**(n_levels-i+2)
            skip_ch = 2 ** (n_levels - i + 2)
            self.fuse.append(
                nn.Sequential(
                    nn.Conv3d(ch + skip_ch, ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch), num_channels=ch, eps=1e-6),
                    nn.GELU(),
                )
            )

            # use SPADE here
            num_spade = min(self.n_spade_blocks, n_res_blocks) # how many SPADE blocks?
            stage = nn.ModuleList()
            for j in range(n_res_blocks):
                if j < num_spade:
                    # first num_spade blocks use SPADE
                    stage.append(ConvNeXtSPADEBlock3D(channels=ch, num_anomaly_classes=num_anomaly_classes, mlp_ratio=4.0))
                else:
                    # normal convnext blocks
                    stage.append(ConvNeXtBlock3D(channels=ch, mlp_ratio=4.0, gn_groups=gn_groups))
            self.blocks.append(stage)

            prev_ch = ch

        self.out = nn.Conv3d(prev_ch, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def set_skips(self, skips: Optional[List[torch.Tensor]]) -> None:
        self._skips = skips

    # take z AND mask
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.from_z(z)

        # use reversed skips: deepest skip is last; None means zero-skips
        skips = self._skips
        if skips is None and len(skips) != self.n_levels:
            raise ValueError(f"Expected {self.n_levels} skips, got {len(skips)}")

        for i in range(self.n_levels):
            x = self.ups[i](x)

            if skips is None:
                skip_ch = 2 ** (self.n_levels - i + 2)
                skip = torch.zeros(
                    (x.shape[0], skip_ch, x.shape[-3], x.shape[-2], x.shape[-1]),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                skip = skips[-1 - i]

            # Skip dropout (training only): forces decoder to use latent z instead of bypassing via skips
            if self.training and self.skip_dropout_p > 0.0:
                skip = F.dropout3d(skip, p=self.skip_dropout_p, training=True)
            if x.shape[-3:] != skip.shape[-3:]:
                # Center-crop the larger one to the smaller
                target = (
                    min(x.shape[-3], skip.shape[-3]),
                    min(x.shape[-2], skip.shape[-2]),
                    min(x.shape[-1], skip.shape[-1]),
                )
                x = _crop_like_3d(x, target)
                skip = _crop_like_3d(skip, target)

            # Skip gating: downscale skip strength to reduce bypass and force latent usage
            skip = skip * self.skip_alpha

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


def _best_gn_groups(default_groups: int, channels: int) -> int:
    """Choose a GroupNorm group count that divides channels."""
    g = min(default_groups, channels)
    while g > 1 and (channels % g != 0):
        g -= 1
    return g


def _upsample_block3d(in_ch: int, out_ch: int, scale: int, use_transpose_conv: bool, gn_groups: int) -> nn.Sequential:
    if use_transpose_conv:
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=scale, stride=scale, padding=0, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, out_ch), num_channels=out_ch, eps=1e-6),
            nn.GELU(),
        )

    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=False),
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, out_ch), num_channels=out_ch, eps=1e-6),
        nn.GELU(),
    )


# -------------------------
# VAE 3D
# -------------------------

@dataclass
class Config:
    """Hyperparameters for ConvNeXtVAE3D.

    Kept identical to the original file for drop-in compatibility.

    Notes:
      - `use_multires_skips` is kept but ignored by the U-Net implementation.
      - `use_transpose_conv` is still honored.
    """
    in_channels: int = None
    num_anomaly_classes: int = None
    n_res_blocks: int = 8
    n_spade_blocks: int = 2 # how many of the res blocks should use spade? for every upscale?
    n_levels: int = 4
    z_channels: int = 250
    bottleneck_dim: int = 250
    use_multires_skips: bool = True
    recon_weight: float = 100.0
    beta_kl: float = 1.0
    beta_kl_start: float = 0.0
    beta_kl_max: float = 4.0
    beta_kl_warmup_start: float = 0
    beta_kl_warmup_epochs: int = 100
    recon_loss: str = "smoothl1"  # 'smoothl1' or 'mse'
    recon_smoothl1_beta: float = 1.0
    use_transpose_conv: bool = True
    fg_weight: float = 1.0
    fg_threshold: float = 0.0

    # Probability for dropping encoder skip features during training (prevents latent bypass in U-Net VAEs)
    # 0.0 disables skip dropout. Typical values: 0.1 - 0.4
    skip_dropout_p: float = 0.0

    # Skip gating factor: scales skip features before concatenation in the decoder.
    # 1.0 disables gating (default). Typical values for encouraging latent usage: 0.2 - 0.6
    skip_alpha: float = 1.0

class ConvNeXtcVAE3D(nn.Module):
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
            use_multires_skips=cfg.use_multires_skips,
        )

        # Decoder reconstructs from latent feature map; skips are set each forward
        self.decoder = ConvNeXtSPADEUNetDecoder3D(
            out_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            n_spade_blocks=cfg.n_spade_blocks,
            z_channels=cfg.z_channels,
            num_anomaly_classes=cfg.num_anomaly_classes,
            use_multires_skips=cfg.use_multires_skips,
            use_transpose_conv=cfg.use_transpose_conv,
            skip_dropout_p=getattr(cfg, 'skip_dropout_p', 0.0),
            skip_alpha=getattr(cfg, 'skip_alpha', 1.0),
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

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z ~ N(mu, sigma^2) using mu + eps*sigma."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        x_pad, pad = _pad_to_multiple_3d(x, multiple)
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
        recon = _crop_like_3d(recon, ref_dhw)
        x_ref = _crop_like_3d(x_pad, ref_dhw) if sum(pad) else x

        return {"recon": recon, "mu": mu, "logvar": logvar, "x_ref": x_ref}

    def loss(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """VAE loss = recon_weight * ReconLoss(recon,x) + beta_kl * KL(mu,logvar)."""
        recon = out["recon"]
        x = out["x_ref"]
        mu, logvar = out["mu"], out["logvar"]

        loss_name = getattr(self.cfg, "recon_loss", "smoothl1").lower()
        if loss_name == "mse":
            recon_per_voxel = (recon - x) ** 2
        else:
            beta = float(getattr(self.cfg, "recon_smoothl1_beta", 1.0))
            try:
                recon_per_voxel = F.smooth_l1_loss(recon, x, reduction="none", beta=beta)
            except TypeError:
                recon_per_voxel = F.smooth_l1_loss(recon, x, reduction="none")

        fg_weight = float(getattr(self.cfg, "fg_weight", 1.0))
        fg_threshold = float(getattr(self.cfg, "fg_threshold", 0.0))
        if fg_weight != 1.0:
            fg_mask = (x > fg_threshold).float()
            weights = torch.where(fg_mask > 0, fg_weight, 1.0)
            recon_loss = (recon_per_voxel * weights).mean()
        else:
            recon_loss = recon_per_voxel.mean()

        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        kl_raw = kl

        recon_weighted = self.cfg.recon_weight * recon_loss
        kl_weighted = self.cfg.beta_kl * kl
        total = recon_weighted + kl_weighted

        return {
            "total": total,
            "recon": recon_loss,
            "kl": kl,
            "kl_raw": kl_raw,
            "recon_weighted": recon_weighted,
            "kl_weighted": kl_weighted,
        }

    def _extract_inputs(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the input tensor x and the mask from a batch (unchanged from original)."""
        # if dataloader loads dict (with keys "img"/"x", "mask"/"ori_mask", "tgt_mask")
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

        # if dataloader loads Tuple[torch.Tensor, torch.Tensor, ...] (like AnomalyDataset)
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
        """Train for one epoch (unchanged API/behavior)."""
        device = torch.device(device)
        model = self.to(device)

        def run_epoch(loader: Iterable, training: bool) -> Dict[str, float]:
            model.train(training)
            run = {"total": 0.0, "recon": 0.0, "kl": 0.0, "kl_raw": 0.0, "recon_weighted": 0.0, "kl_weighted": 0.0}
            n = 0

            pbar = tqdm(loader, desc=("train" if training else "val"), leave=False, dynamic_ncols=True)
            for step, batch in enumerate(pbar, start=1):
                x, ori_mask, _ = self._extract_inputs(batch=batch)  # no need for tgt_mask in training
                x = x.to(device, non_blocking=True)
                ori_mask = ori_mask.to(device, non_blocking=True)

                if x.ndim != 5 or ori_mask.ndim not in [4, 5]:
                    raise ValueError(f"Expected (B,C,D,H,W) from dataloader, got {tuple(x.shape)} (x) and "
                                     f"{tuple(ori_mask.shape)} (ori_mask).")

                if training:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    # use ori_mask as tgt_mask in training (so that SPADE is not contradicting the reconstruction score)
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
        target_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,  # use target mask if you want spatial variation
        *,
        n: int = 1,
        s: float = 0.8,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Generate *n* slightly varied variants around a given sample.

        This performs *posterior sampling*:
            z = mu + s * sigma * eps,  eps ~ N(0, I)

        Meaning of the parameters:
          - n: how many variants to generate per input sample.
          - s: "strength" / "temperature" of the variation.
               s=0.0 -> deterministic reconstruction (uses mu only)
               s~0.2-0.5 -> small variations (recommended)
               s>=1.0 -> large variations (can drift away from the input)

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
        
        # --- WICHTIG: KEIN .float() mehr für die Masken! ---
        ori_mask = torch.as_tensor(original_mask)
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
            x_pad, pad = _pad_to_multiple_3d(x, multiple)
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
            if s == 0.0:
                z = mu.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
            else:
                eps = torch.randn((B, n, mu.shape[-1]), device=device, dtype=mu.dtype)
                z = (mu.unsqueeze(1) + (s * std).unsqueeze(1) * eps).reshape(B * n, -1)

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
            recon = _crop_like_3d(recon, ref_dhw)

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

    def warmup(self, shape, device=None, dtype=None):
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
    
    def generate_synth_sample_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        *,
        s: float = 1.0,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate synthetic 3D samples via *prior sampling* conditioned on a target mask.
        (No input image required).

        Samples:
            z ~ N(0, I) (scaled by s), then decode to 3D volume space conditioned on target_mask.

        Parameters:
        - sample: sample dict containing "tgt_mask", or a label mask with
          shape (D,H,W), (C,D,H,W) or (B,C,D,H,W). Defines the classes and output spatial size.
        - s: prior temperature / diversity strength (1.0 is standard; <1.0 more conservative).
        - clamp_01: clamp outputs to [0,1].
        - return_torch: return torch.Tensor instead of np.ndarray.

        Output:
        - (C, D, H, W) if input was 4D
        - (B, C, D, H, W) if input was 5D
        """
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        model.decoder.set_skips(None)

        if isinstance(sample, dict):
            target_mask = sample.get("tgt_mask", sample.get("ori_mask", sample.get("mask")))
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
            tgt_mask_pad, pad = _pad_to_multiple_3d(tgt_mask_oh, multiple)

            # Calculate latent dimensions based on the padded mask
            latent_dhw = (
                tgt_mask_pad.shape[2] // multiple,
                tgt_mask_pad.shape[3] // multiple,
                tgt_mask_pad.shape[4] // multiple
            )
            
            model._ensure_fcs(latent_dhw, device)

            B = tgt_mask_oh.shape[0]
            z_dim = int(getattr(self.cfg, "bottleneck_dim", 256))

            # Prior sampling: z ~ N(0, I)
            if s == 0.0:
                z = torch.zeros((B, z_dim), device=device)
            else:
                z = torch.randn((B, z_dim), device=device) * float(s)

            # Map z -> decoder feature map
            h_dec = model.fc_decode(z).reshape(B, int(self.cfg.z_channels), *latent_dhw)

            # Decode conditioned on the target mask
            recon = model.decoder(h_dec, tgt_mask_pad)
            
            # Crop back to exact requested size
            recon = _crop_like_3d(recon, ref_dhw)

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
    
if __name__ == "__main__":
    # Quick sanity check
    cfg = Config(in_channels=1, num_anomaly_classes=4, n_res_blocks=2, n_levels=4, z_channels=64, bottleneck_dim=64)  # 4 classes one-hot encoded
    model = ConvNeXtcVAE3D(cfg=cfg)
    
    x = torch.randn(1, 1, 64, 64, 64)
    mask = torch.zeros(1, 64, 64, 64, dtype=torch.long)
    
    out = model(x, mask)
    print({k: tuple(v.shape) for k, v in out.items()})
    