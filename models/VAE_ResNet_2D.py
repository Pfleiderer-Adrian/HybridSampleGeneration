from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------
# helpers: padding/cropping (2D only)
# -------------------------
def _compute_symmetric_pad(size: int, multiple: int) -> Tuple[int, int]:
    """
    Compute symmetric (left,right) padding so that `size` becomes divisible by `multiple`.
    """
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
    """
    Pad a 4D tensor (B, C, H, W) so that H/W are divisible by `multiple`.

    PyTorch F.pad order for 4D is: (wL, wR, hL, hR)

    Returns:
      - x_padded: (B,C,H',W')
      - pad: (wL,wR,hL,hR)
    """
    h, w = x.shape[-2:]

    hL, hR = _compute_symmetric_pad(h, multiple)
    wL, wR = _compute_symmetric_pad(w, multiple)

    pad = (wL, wR, hL, hR)
    if sum(pad) == 0:
        return x, pad

    return F.pad(x, pad, mode="constant", value=0.0), pad


def _crop_like_2d(x: torch.Tensor, ref_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Center-crop a tensor spatially to match a reference size (H,W).
    Works for tensors shaped (..., H, W).
    """
    h_ref, w_ref = ref_hw
    h, w = x.shape[-2:]

    def sl(cur: int, ref: int):
        if cur == ref:
            return slice(None)
        start = (cur - ref) // 2
        return slice(start, start + ref)

    return x[..., sl(h, h_ref), sl(w, w_ref)]


# -------------------------
# blocks (2D only)
# -------------------------
class ResidualBlock2D(nn.Module):
    """
    Basic residual block for 2D images.

    Input:  (B, in_ch, H, W)
    Output: (B, out_ch, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int, leak: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(leak, inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(leak, inplace=True)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        return identity + out


class ResNetEncoder2D(nn.Module):
    """
    ResNet-style 2D encoder with optional multi-resolution skip aggregation.

    Input:  x (B, C, H, W)
    Output: h (B, z_channels, h', w')  with spatial dims reduced by 2**n_levels
    """
    def __init__(
        self,
        in_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_multires_skips: bool = True,
        leak: float = 0.2,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.use_multires_skips = use_multires_skips
        self.max_filters = 2 ** (n_levels + 3)

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(leak, inplace=True),
        )

        self.res_stages = nn.ModuleList()
        self.down_stages = nn.ModuleList()
        self.skip_stages = nn.ModuleList()

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)

            self.res_stages.append(
                nn.Sequential(
                    *[ResidualBlock2D(n_filters_1, n_filters_1, leak=leak) for _ in range(n_res_blocks)]
                )
            )

            self.down_stages.append(
                nn.Sequential(
                    nn.Conv2d(n_filters_1, n_filters_2, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(n_filters_2),
                    nn.LeakyReLU(leak, inplace=True),
                )
            )

            if use_multires_skips:
                ks = 2 ** (n_levels - i)
                self.skip_stages.append(
                    nn.Sequential(
                        nn.Conv2d(n_filters_1, self.max_filters, kernel_size=ks, stride=ks, padding=0, bias=False),
                        nn.BatchNorm2d(self.max_filters),
                        nn.LeakyReLU(leak, inplace=True),
                    )
                )

        self.output_conv = nn.Conv2d(2 ** (n_levels + 3), z_channels, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_stages[i](x)
            if self.use_multires_skips:
                skips.append(self.skip_stages[i](x))
            x = self.down_stages[i](x)

        if self.use_multires_skips:
            x = x + torch.stack(skips, dim=0).sum(dim=0)

        return self.output_conv(x)


class ResNetDecoder2D(nn.Module):
    """
    ResNet-style 2D decoder with optional multi-resolution skip injections from the top latent.

    Input:  z (B, z_channels, h', w')
    Output: (B, out_channels, H, W) after upsampling
    """
    def __init__(
        self,
        out_channels: int,
        n_res_blocks: int,
        n_levels: int,
        z_channels: int,
        use_multires_skips: bool = True,
        leak: float = 0.2,
        use_transpose_conv: bool = True,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.use_multires_skips = use_multires_skips
        self.max_filters = 2 ** (n_levels + 3)
        self.use_transpose_conv = use_transpose_conv

        self.input_conv = nn.Sequential(
            nn.Conv2d(z_channels, self.max_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.max_filters),
            nn.LeakyReLU(leak, inplace=True),
        )

        self.up_stages = nn.ModuleList()
        self.res_stages = nn.ModuleList()
        self.skip_stages = nn.ModuleList()

        def upsample_block(in_ch: int, out_ch: int, scale: int) -> nn.Sequential:
            if self.use_transpose_conv:
                return nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale, stride=scale, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(leak, inplace=True),
                )
            return nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(leak, inplace=True),
            )

        prev_ch = self.max_filters
        for i in range(n_levels):
            n_filters = 2 ** (self.n_levels - i + 2)

            self.up_stages.append(
                upsample_block(prev_ch, n_filters, scale=2)
            )
            prev_ch = n_filters

            self.res_stages.append(
                nn.Sequential(
                    *[ResidualBlock2D(n_filters, n_filters, leak=leak) for _ in range(n_res_blocks)]
                )
            )

            if use_multires_skips:
                ks = 2 ** (i + 1)
                self.skip_stages.append(
                    upsample_block(self.max_filters, n_filters, scale=ks)
                )

        self.output_conv = nn.Conv2d(prev_ch, out_channels, 3, 1, 1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.up_stages[i](z)
            z = self.res_stages[i](z)
            if self.use_multires_skips:
                z = z + self.skip_stages[i](z_top)

        return self.output_conv(z)


# -------------------------
# VAE 2D
# -------------------------
@dataclass
class Config:
    """
    Hyperparameters for ResNetVAE2D.

    Mirrors the 3D config for easier swapping.
    """
    n_res_blocks: int = 8
    n_levels: int = 4
    z_channels: int = 250
    bottleneck_dim: int = 250
    use_multires_skips: bool = True
    recon_weight: float = 100.0
    beta_kl: float = 1.0

    # --- continuous-intensity reconstruction (Option B) ---
    # Default: SmoothL1 (Huber) is typically more robust for MRI intensities than BCE.
    # Supported: "smoothl1" | "mse"
    recon_loss: str = "smoothl1"
    # Only used when recon_loss == "smoothl1". (PyTorch calls this parameter "beta".)
    recon_smoothl1_beta: float = 1.0
    # If True, decoder uses ConvTranspose2d for upsampling. If False, uses Upsample(bilinear)+Conv2d.
    use_transpose_conv: bool = True
    use_transpose_conv: bool = True


class ResNetVAE2D(nn.Module):
    """
    2D ResNet-VAE.

    Expected input:
      - x: (B, C, H, W), float (continuous intensities; e.g. MRI)
      - No implicit clamping to [0,1]. If you want standardization/normalization,
        do it in your dataset/pipeline (recommended: z-score or robust scaling).

    Forward output:
      - recon: (B,C,H,W)
      - mu/logvar: (B,bottleneck_dim)
      - x_ref: (B,C,H,W) reference input for recon loss
    """
    def __init__(self, in_channels: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        self.encoder = ResNetEncoder2D(
            in_channels=in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_multires_skips=cfg.use_multires_skips,
        )

        self.decoder = ResNetDecoder2D(
            out_channels=in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_multires_skips=cfg.use_multires_skips,
            use_transpose_conv=cfg.use_transpose_conv,
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

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[1]}")

        # Continuous-valued inputs: keep intensities as-is (no auto-normalization / clamping).
        x = x.float()

        device = x.device
        B = x.shape[0]
        ref_hw = tuple(x.shape[-2:])

        multiple = 2 ** self.cfg.n_levels
        x_pad, pad = _pad_to_multiple_2d(x, multiple)

        h = self.encoder(x_pad)  # (B, z_channels, h', w')
        latent_hw = tuple(h.shape[-2:])

        self._ensure_fcs(latent_hw, device)

        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        z = self.reparameterize(mu, logvar)

        h_dec = self.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_hw)
        # Linear reconstruction head (no sigmoid) for continuous intensities.
        recon = self.decoder(h_dec)

        recon = _crop_like_2d(recon, ref_hw)
        x_ref = _crop_like_2d(x_pad, ref_hw) if sum(pad) else x

        return {"recon": recon, "mu": mu, "logvar": logvar, "x_ref": x_ref}

    def loss(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        recon = out["recon"]
        x = out["x_ref"]
        mu, logvar = out["mu"], out["logvar"]

        # Continuous-intensity reconstruction loss
        loss_name = str(getattr(self.cfg, "recon_loss", "smoothl1")).lower()
        if loss_name in ("smoothl1", "huber", "smooth_l1"):
            beta = float(getattr(self.cfg, "recon_smoothl1_beta", 1.0))
            try:
                recon_loss = F.smooth_l1_loss(recon, x, reduction="mean", beta=beta)
            except TypeError:
                # Older PyTorch without beta parameter
                recon_loss = F.smooth_l1_loss(recon, x, reduction="mean")
        elif loss_name in ("mse", "l2"):
            recon_loss = F.mse_loss(recon, x, reduction="mean")
        else:
            raise ValueError(
                f"Unknown cfg.recon_loss={self.cfg.recon_loss!r}. Supported: 'smoothl1' | 'mse'"
            )
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        total = self.cfg.recon_weight * recon_loss + self.cfg.beta_kl * kl
        return {"total": total, "recon": recon_loss, "kl": kl}

    def _extract_x(self, batch) -> torch.Tensor:
        # Special case: list with exactly 2 elements and first is tensor
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
        device = torch.device(device)
        model = self.to(device)

        def run_epoch(loader: Iterable, training: bool) -> Dict[str, float]:
            model.train(training)
            run = {"total": 0.0, "recon": 0.0, "kl": 0.0}
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
        sample: np.ndarray,
        *,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
    ) -> np.ndarray:
        """
        Generate a synthetic sample (reconstruction) for a SINGLE input sample.

        Accepted input shapes:
          - (C, H, W)   (single sample)
          - (1, C, H, W) (batched single sample)

        Returns:
          - (C, H, W) float32
        """
        device = torch.device(device)
        model = self.to(device)
        model.eval()

        if not isinstance(sample, torch.Tensor):
            x = torch.as_tensor(sample)
        else:
            x = sample

        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1,C,H,W)
        if x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D input, got {tuple(x.shape)}")

        x = x.float()

        # Continuous-valued inputs: keep intensities as-is.
        # (Optional) For legacy pipelines you can set clamp_01=True.

        if clamp_01:
            x = x.clamp(0.0, 1.0)

        x = x.to(device)

        with torch.no_grad():
            out = model(x)
            recon = out["recon"]
            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

        recon_np = recon.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
        return recon_np

    def warmup(self, shape, device=None, dtype=None):
        """
        Warm up the model to initialize lazy FC layers.

        shape: (C, H, W)
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
    cfg = Config(n_res_blocks=4, n_levels=4, z_channels=128, bottleneck_dim=128)
    model = ResNetVAE2D(in_channels=1, cfg=cfg)
