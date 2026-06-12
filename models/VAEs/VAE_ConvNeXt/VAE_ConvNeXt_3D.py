from __future__ import annotations

"""ConvNeXt3D-U-Net VAE

Features:
- ConvNeXt3D blocks (depthwise conv + pointwise MLP)
- BatchNorm -> GroupNorm (more stable for 3D and small batch sizes)
- True U-Net skip connections (feature concatenation)

"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, Union, List
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.model_inferface import HybridModelInterface


def _create_tgt_mask_from_synth_anomaly(synth_anomaly_image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if torch.is_tensor(synth_anomaly_image):
        background_threshold = torch.nanmin(synth_anomaly_image) + 0.001
        synth_projection = torch.amax(synth_anomaly_image, dim=0)
        return (synth_projection > background_threshold).to(torch.uint8)

    background_threshold = float(np.nanmin(synth_anomaly_image)) + 0.001
    synth_projection = np.max(synth_anomaly_image, axis=0)
    return (synth_projection > background_threshold).astype(np.uint8)


# -------------------------
# blocks (3D)
# -------------------------

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


class ConvNeXtUNetDecoder3D(nn.Module):
    """ConvNeXt3D decoder with U-Net skips.

    For API compatibility with the original decoder, forward(z) only takes `z`.
    Skips are provided via set_skips(skips) before calling forward. If skips
    are set to None, the decoder uses zero-skips for pure prior sampling.
    """

    def __init__(
        self,
        out_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
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

            stage = []
            for _ in range(n_res_blocks):
                stage.append(ConvNeXtBlock3D(ch, mlp_ratio=4.0, gn_groups=gn_groups))
            self.blocks.append(nn.Sequential(*stage))

            prev_ch = ch

        self.out = nn.Conv3d(prev_ch, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def set_skips(self, skips: Optional[List[torch.Tensor]]) -> None:
        self._skips = skips

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        x = self.from_z(z)

        # use reversed skips: deepest skip is last; None means zero-skips
        skips = self._skips
        if skips is not None and len(skips) != self.n_levels:
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
                x = HybridModelInterface._crop_like(x, target)
                skip = HybridModelInterface._crop_like(skip, target)

            # Skip gating: downscale skip strength to reduce bypass and force latent usage
            skip = skip * self.skip_alpha

            x = torch.cat([x, skip], dim=1)
            x = self.fuse[i](x)
            x = self.blocks[i](x)

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
    n_res_blocks: int = 8
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


class ConvNeXtVAE3D(HybridModelInterface):
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
            use_multires_skips=cfg.use_multires_skips,
        )

        # Decoder reconstructs from latent feature map; skips are set each forward
        self.decoder = ConvNeXtUNetDecoder3D(
            out_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
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
        x_pad, pad = HybridModelInterface._pad_to_multiple(x, multiple)

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
        recon = HybridModelInterface._crop_like(recon, ref_dhw)
        x_ref = HybridModelInterface._crop_like(x_pad, ref_dhw) if sum(pad) else x

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
                x = self._extract_x(batch=batch)
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                x = x.to(device, non_blocking=True)

                if x.ndim != 5:
                    raise ValueError(f"Expected (B,C,D,H,W) from dataloader, got {tuple(x.shape)}")

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
        sample: Union[dict, np.ndarray, torch.Tensor],
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
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")

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
            x_pad, pad = HybridModelInterface._pad_to_multiple(x, multiple)

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

            recon = model.decoder(h_dec)
            recon = HybridModelInterface._crop_like(recon, ref_dhw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            # Reshape back to (B,n,C,D,H,W)
            recon = recon.view(B, n, self.in_channels, *ref_dhw)

            if single:
                recon = recon.squeeze(0)  # (n,C,D,H,W)
                recon = recon.squeeze(0)

        if return_torch:
            return recon, _create_tgt_mask_from_synth_anomaly(recon)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, _create_tgt_mask_from_synth_anomaly(recon_np)

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
            _ = self(x)

        if was_training:
            self.train()

        return self
    
    def generate_synth_sample_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor, None] = None,
        *,
        out_dhw: tuple[int, int, int] | None = None,
        s: float = 0.5,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Generate ONE synthetic 3D sample via *prior sampling* (no input sample required).

        Samples:
            z ~ N(0, I)  (scaled by s), then decode to 3D volume space.

        Parameters:
        - out_dhw: output (D, H, W). If None, tries cfg.sample_dhw or cfg.image_dhw, else defaults to (64, 64, 64).
        - s: prior temperature / diversity strength (1.0 is standard; <1.0 more conservative; >1.0 more diverse).
        - clamp_01: clamp outputs to [0,1].
        - return_torch: return torch.Tensor instead of np.ndarray.

        Output:
        - (C, D, H, W)
        """
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")

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
        z_dim = int(getattr(self.cfg, "bottleneck_dim", 256))

        with torch.no_grad():
            model._ensure_fcs(latent_dhw, device)

            # Prior sampling: z ~ N(0, I)
            if s == 0.0:
                z = torch.zeros((1, z_dim), device=device)
            else:
                z = torch.randn((1, z_dim), device=device) * float(s)

            # Map z -> decoder feature map and decode
            h_dec = model.fc_decode(z).reshape(1, int(self.cfg.z_channels), *latent_dhw)
            recon = model.decoder(h_dec)  # (1, C, D_pad, H_pad, W_pad) typically

            # Crop back to requested size and drop batch dim
            recon = recon[..., :D, :H, :W].squeeze(0)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

        if return_torch:
            return recon, _create_tgt_mask_from_synth_anomaly(recon)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, _create_tgt_mask_from_synth_anomaly(recon_np)


if __name__ == "__main__":
    # Quick sanity check
    cfg = Config(n_res_blocks=2, n_levels=4, z_channels=64, bottleneck_dim=64)
    model = ConvNeXtVAE3D(in_channels=1, cfg=cfg)
    x = torch.randn(1, 1, 64, 64, 64)
    out = model(x)
    print({k: tuple(v.shape) for k, v in out.items()})
