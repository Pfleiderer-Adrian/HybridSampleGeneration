from __future__ import annotations

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

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, Union, List
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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


def _pad_to_multiple_2d(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Pad (B,C,H,W) so H/W are divisible by `multiple` (symmetric, constant=0)."""
    h, w = x.shape[-2:]

    hL, hR = _compute_symmetric_pad(h, multiple)
    wL, wR = _compute_symmetric_pad(w, multiple)

    pad = (wL, wR, hL, hR)
    if sum(pad) == 0:
        return x, pad

    return F.pad(x, pad, mode="constant", value=0.0), pad


def _crop_like_2d(x: torch.Tensor, ref_hw: Tuple[int, int]) -> torch.Tensor:
    """Center-crop last 2 dims of x to ref_hw."""
    h_ref, w_ref = ref_hw
    h, w = x.shape[-2:]

    def sl(cur: int, ref: int):
        if cur == ref:
            return slice(None)
        start = (cur - ref) // 2
        return slice(start, start + ref)

    return x[..., sl(h, h_ref), sl(w, w_ref)]


# -------------------------
# blocks (2D)
# -------------------------

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
    """ConvNeXt-style block for 2D images (channels-first).

    Design:
      depthwise conv (k=7 in classic ConvNeXt, here k=7) -> GroupNorm
      -> pointwise expand -> GELU -> pointwise project
      + residual

    Notes:
      - GroupNorm used to avoid channel-last permutations.
      - Capacity can be scaled via mlp_ratio.
    """

    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        gn_groups: int = 8,
        drop_path: float = 0.0,
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
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

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
    """ConvNeXt2D encoder with U-Net skip outputs.

    forward(x) returns:
      - h: deepest latent feature map (B, z_channels, h', w')
      - skips: list of feature maps at each resolution (for decoder), length n_levels

    Channel schedule:
      base=8 -> 16 -> 32 -> 64 -> ...
    """

    def __init__(
        self,
        in_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_multires_skips: bool = True,  # kept for API compatibility (not used)
        leak: float = 0.2,               # kept for API compatibility
        gn_groups: int = 8,
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

        for i in range(n_levels):
            ch = 2 ** (i + 3)       # 8, 16, 32, 64...
            ch_next = 2 ** (i + 4)  # 16, 32, 64, 128...

            stage = []
            for _ in range(n_res_blocks):
                stage.append(ConvNeXtBlock2D(ch, mlp_ratio=4.0, gn_groups=gn_groups))
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


class ConvNeXtUNetDecoder2D(nn.Module):
    """ConvNeXt2D decoder with U-Net skips.

    For API compatibility, forward(z) only takes `z`.
    Skips must be set via set_skips(skips) before forward.
    """

    def __init__(
        self,
        out_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_multires_skips: bool = True,  # kept for API compatibility (not used)
        leak: float = 0.2,               # kept for API compatibility
        use_transpose_conv: bool = True,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.use_transpose_conv = use_transpose_conv
        self._skips: Optional[List[torch.Tensor]] = None

        self.bottom_ch = 2 ** (n_levels + 3)
        self.from_z = nn.Sequential(
            nn.Conv2d(z_channels, self.bottom_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, self.bottom_ch), num_channels=self.bottom_ch, eps=1e-6),
            nn.GELU(),
        )

        self.ups: nn.ModuleList = nn.ModuleList()
        self.fuse: nn.ModuleList = nn.ModuleList()
        self.blocks: nn.ModuleList = nn.ModuleList()

        prev_ch = self.bottom_ch
        for i in range(n_levels):
            # mirror channel schedule of encoder
            ch = 2 ** (n_levels - i + 2)

            self.ups.append(_upsample_block2d(prev_ch, ch, scale=2, use_transpose_conv=use_transpose_conv, gn_groups=gn_groups))

            # skip channels at matching resolution
            skip_ch = 2 ** (n_levels - i + 2)
            self.fuse.append(
                nn.Sequential(
                    nn.Conv2d(ch + skip_ch, ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GroupNorm(num_groups=_best_gn_groups(gn_groups, ch), num_channels=ch, eps=1e-6),
                    nn.GELU(),
                )
            )

            stage = []
            for _ in range(n_res_blocks):
                stage.append(ConvNeXtBlock2D(ch, mlp_ratio=4.0, gn_groups=gn_groups))
            self.blocks.append(nn.Sequential(*stage))

            prev_ch = ch

        self.out = nn.Conv2d(prev_ch, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def set_skips(self, skips: List[torch.Tensor]) -> None:
        self._skips = skips

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self._skips is None:
            raise RuntimeError("Decoder skips are not set. Call decoder.set_skips(skips) before forward().")

        x = self.from_z(z)

        skips = self._skips
        if len(skips) != self.n_levels:
            raise ValueError(f"Expected {self.n_levels} skips, got {len(skips)}")

        for i in range(self.n_levels):
            x = self.ups[i](x)

            skip = skips[-1 - i]

            # Align spatial sizes (off-by-1 for odd inputs)
            if x.shape[-2:] != skip.shape[-2:]:
                target = (min(x.shape[-2], skip.shape[-2]), min(x.shape[-1], skip.shape[-1]))
                x = _crop_like_2d(x, target)
                skip = _crop_like_2d(skip, target)

            x = torch.cat([x, skip], dim=1)
            x = self.fuse[i](x)
            x = self.blocks[i](x)

        return self.out(x)


# -------------------------
# VAE 2D
# -------------------------

@dataclass
class Config:
    """Hyperparameters for ConvNeXtVAE2D.

    Notes:
      - use_multires_skips is kept but ignored by the U-Net implementation.
      - use_transpose_conv is honored.
    """

    n_res_blocks: int = 8
    n_levels: int = 4
    z_channels: int = 250
    bottleneck_dim: int = 250
    use_multires_skips: bool = True
    recon_weight: float = 100.0
    beta_kl: float = 1.0
    recon_loss: str = "smoothl1"  # 'smoothl1' or 'mse'
    recon_smoothl1_beta: float = 1.0
    use_transpose_conv: bool = True
    fg_weight: float = 1.0
    fg_threshold: float = 0.0


class ConvNeXtVAE2D(nn.Module):
    """2D ConvNeXt-U-Net VAE.

    Expected input:
      - x: (B, C, H, W)

    Forward output dict:
      - recon: reconstructed x (B,C,H,W)
      - mu: mean vector (B, bottleneck_dim)
      - logvar: log-variance vector (B, bottleneck_dim)
      - x_ref: reference input used for reconstruction loss (cropped/padded version)
    """

    def __init__(self, in_channels: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        self.encoder = ConvNeXtUNetEncoder2D(
            in_channels=in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_multires_skips=cfg.use_multires_skips,
        )

        self.decoder = ConvNeXtUNetDecoder2D(
            out_channels=in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_multires_skips=cfg.use_multires_skips,
            use_transpose_conv=cfg.use_transpose_conv,
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

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z ~ N(mu, sigma^2) using mu + eps*sigma."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder -> bottleneck -> decoder."""
        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[1]}")

        x = x.float()
        device = x.device
        B = x.shape[0]
        ref_hw = tuple(x.shape[-2:])

        multiple = 2 ** self.cfg.n_levels
        x_pad, pad = _pad_to_multiple_2d(x, multiple)

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

        recon = _crop_like_2d(recon, ref_hw)
        x_ref = _crop_like_2d(x_pad, ref_hw) if sum(pad) else x

        return {"recon": recon, "mu": mu, "logvar": logvar, "x_ref": x_ref}

    def loss(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """VAE loss = recon_weight * ReconLoss(recon,x) + beta_kl * KL(mu,logvar)."""
        recon = out["recon"]
        x = out["x_ref"]
        mu, logvar = out["mu"], out["logvar"]

        loss_name = getattr(self.cfg, "recon_loss", "smoothl1").lower()
        if loss_name == "mse":
            recon_per_pixel = (recon - x) ** 2
        else:
            beta = float(getattr(self.cfg, "recon_smoothl1_beta", 1.0))
            try:
                recon_per_pixel = F.smooth_l1_loss(recon, x, reduction="none", beta=beta)
            except TypeError:
                recon_per_pixel = F.smooth_l1_loss(recon, x, reduction="none")

        fg_weight = float(getattr(self.cfg, "fg_weight", 1.0))
        fg_threshold = float(getattr(self.cfg, "fg_threshold", 0.0))
        if fg_weight != 1.0:
            fg_mask = (x > fg_threshold).float()
            weights = torch.where(fg_mask > 0, fg_weight, 1.0)
            recon_loss = (recon_per_pixel * weights).mean()
        else:
            recon_loss = recon_per_pixel.mean()

        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        recon_weighted = self.cfg.recon_weight * recon_loss
        kl_weighted = self.cfg.beta_kl * kl
        total = recon_weighted + kl_weighted

        return {
            "total": total,
            "recon": recon_loss,
            "kl": kl,
            "recon_weighted": recon_weighted,
            "kl_weighted": kl_weighted,
        }

    def _extract_x(self, batch) -> torch.Tensor:
        """Extract the input tensor x from a batch (kept compatible with the template)."""
        if isinstance(batch, list) and len(batch) == 2:
            if isinstance(batch[0], torch.Tensor):
                return batch[0]

        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            x = batch[0]
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)

        if isinstance(batch, dict):
            for key in ("x", "image", "inputs"):
                if key in batch:
                    v = batch[key]
                    if isinstance(v, torch.Tensor):
                        return v
                    return torch.as_tensor(v)

        raise TypeError(f"Unknown batch type: {type(batch)}")

    def fit_epoch(
        self,
        train_dataloader,
        val_dataloader,
        optimizer,
        *,
        log_every=1,
        grad_clip_norm: Optional[float] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> tuple[dict, dict]:
        """Train for one epoch (same API/behavior as the template)."""
        device = torch.device(device)
        model = self.to(device)

        def run_epoch(loader: Iterable, training: bool) -> Dict[str, float]:
            model.train(training)
            run = {"total": 0.0, "recon": 0.0, "kl": 0.0, "recon_weighted": 0.0, "kl_weighted": 0.0}
            n = 0

            pbar = tqdm(loader, desc=("train" if training else "val"), leave=False, dynamic_ncols=True)
            for step, batch in enumerate(pbar, start=1):
                x = self._extract_x(batch=batch)
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                x = x.to(device, non_blocking=True)

                if x.ndim != 4:
                    raise ValueError(f"Expected (B,C,H,W) from dataloader, got {tuple(x.shape)}")

                if training:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    out = model(x)
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
        sample: Union[np.ndarray, torch.Tensor],
        *,
        n: int = 1,
        s: float = 0.1,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Generate *n* slightly varied variants around a given sample.

        This performs posterior sampling:
            z = mu + s * sigma * eps,  eps ~ N(0, I)

        Parameters:
          - n: number of variants per input sample.
          - s: strength / temperature of the variation.
               s=0.0 -> deterministic reconstruction (uses mu only)
               s~0.2-0.5 -> small variations (recommended)
               s>=1.0 -> large variations (can drift away)

        Input:
          - sample: (C,H,W) or (B,C,H,W)

        Output:
          - if input is (C,H,W): (n,C,H,W)
          - if input is (B,C,H,W): (B,n,C,H,W)
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        if not isinstance(sample, torch.Tensor):
            x = torch.as_tensor(sample)
        else:
            x = sample

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

        with torch.no_grad():
            ref_hw = tuple(x.shape[-2:])
            multiple = 2 ** self.cfg.n_levels
            x_pad, pad = _pad_to_multiple_2d(x, multiple)

            h, skips = model.encoder(x_pad)
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

            rep_skips: List[torch.Tensor] = []
            for sk in skips:
                rep_skips.append(sk.repeat_interleave(n, dim=0))
            model.decoder.set_skips(rep_skips)

            recon = model.decoder(h_dec)
            recon = _crop_like_2d(recon, ref_hw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            recon = recon.view(B, n, self.in_channels, *ref_hw)

            if single:
                recon = recon.squeeze(0)  # (n,C,H,W)
                recon = recon.squeeze(0)  # (C,H,W)

        if return_torch:
            return recon
        return recon.detach().cpu().numpy().astype(np.float32, copy=False)


    def warmup(self, shape, device=None, dtype=None):
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


if __name__ == "__main__":
    # Quick sanity check
    cfg = Config(n_res_blocks=2, n_levels=4, z_channels=64, bottleneck_dim=64)
    model = ConvNeXtVAE2D(in_channels=1, cfg=cfg)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print({k: tuple(v.shape) for k, v in out.items()})

    # Posterior sampling: generate 5 variants per item
    variants = model.generate_synth_sample(x[0], n=5, s=0.3, return_torch=True)
    print("variants:", tuple(variants.shape))
