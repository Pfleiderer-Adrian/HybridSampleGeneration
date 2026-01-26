"""
defect_inpaint_ldm_lora_with_vae.py

Latent Diffusion Inpainting + per-class LoRA adapters, INCLUDING an internal configurable VAE.

Usecase:
- Inputs are images with shape (B, C, H, W) and a BINARY mask (B, 1, H, W)
- You train a base inpainting diffusion model once (class-agnostic),
  then freeze base and train one LoRA per defect class.

Key features:
- Fully self-contained PyTorch code (no diffusers dependency)
- VAE is configurable via ModelConfig (latent channels / downsample factor etc.)
- VAE-like API stays: encode(), decode(), forward()
- Only adds a `class_name` parameter where needed (LoRA selection)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable, Union, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Utils
# --------------------------

def exists(x) -> bool:
    return x is not None


def default(val, d):
    return val if exists(val) else d


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def downsample_mask(mask: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize mask to latent spatial size."""
    if mask.dtype != torch.float32:
        mask = mask.float()
    return F.interpolate(mask, size=target_hw, mode="nearest")


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract t-indexed values from a 1D tensor a into shape compatible with x."""
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# --------------------------
# Flat ModelConfig (no nested configs)
# --------------------------

@dataclass
class ModelConfig:
    # --- VAE ---
    in_channels: int = 3
    vae_base_channels: int = 64
    latent_channels: int = 4
    vae_channel_mults: Tuple[int, ...] = (1, 2, 4, 4)  # controls downsample factor
    vae_num_res_blocks: int = 1
    vae_dropout: float = 0.0

    vae_sample_latent: bool = False      # False => use mu (deterministic)
    vae_latent_scale: float = 1.0        # optional scaling constant
    vae_out_tanh: bool = True            # output in [-1,1] if True

    # --- Diffusion schedule ---
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # --- UNet ---
    unet_base_channels: int = 128
    unet_channel_mults: Tuple[int, ...] = (1, 2, 2, 4)
    unet_res_blocks: int = 2
    unet_time_dim: int = 512
    unet_dropout: float = 0.0

    # --- Loss weights ---
    mask_loss_weight: float = 1.0
    context_loss_weight: float = 0.2

    # --- Sampling ---
    ddim_steps: int = 50
    eta: float = 0.0

    # --- LoRA ---
    lora_rank: int = 8
    lora_alpha: float = 8.0


# --------------------------
# VAE blocks
# --------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ConvVAE(nn.Module):
    """
    A lightweight configurable VAE suitable as the latent bridge for latent diffusion.

    Input:  x  (B, C, H, W)
    Latent: z  (B, latent_channels, H/2^n, W/2^n) where n = number of downsamples
    Output: x' (B, C, H, W)

    encode() returns z (mu or sampled).
    decode() maps z back to image.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        base = cfg.vae_base_channels
        self.enc_in = nn.Conv2d(cfg.in_channels, base, 3, padding=1)

        # Encoder
        enc = []
        ch = base
        for i, mult in enumerate(cfg.vae_channel_mults):
            out_ch = base * mult
            for _ in range(cfg.vae_num_res_blocks):
                enc.append(ResBlock(ch, out_ch, dropout=cfg.vae_dropout))
                ch = out_ch
            if i != len(cfg.vae_channel_mults) - 1:
                enc.append(Downsample(ch))
        self.encoder = nn.Sequential(*enc)

        self.to_mu = nn.Conv2d(ch, cfg.latent_channels, 1)
        self.to_logvar = nn.Conv2d(ch, cfg.latent_channels, 1)

        # Decoder
        self.dec_in = nn.Conv2d(cfg.latent_channels, ch, 3, padding=1)

        dec = []
        for i, mult in reversed(list(enumerate(cfg.vae_channel_mults))):
            out_ch = base * mult
            for _ in range(cfg.vae_num_res_blocks):
                dec.append(ResBlock(ch, out_ch, dropout=cfg.vae_dropout))
                ch = out_ch
            if i != 0:
                dec.append(Upsample(ch))
        self.decoder = nn.Sequential(*dec)

        self.dec_out = nn.Conv2d(ch, cfg.in_channels, 3, padding=1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_in(x)
        h = self.encoder(h)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode_stats(x)
        z = self.reparameterize(mu, logvar) if self.cfg.vae_sample_latent else mu
        return z * self.cfg.vae_latent_scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.cfg.vae_latent_scale
        h = self.dec_in(z)
        h = self.decoder(h)
        x_hat = self.dec_out(h)
        return torch.tanh(x_hat) if self.cfg.vae_out_tanh else x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = self.reparameterize(mu, logvar) if self.cfg.vae_sample_latent else mu
        z = z * self.cfg.vae_latent_scale
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# --------------------------
# Diffusion UNet blocks
# --------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.float32:
            t = t.float()
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class UNetResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    """
    Minimal UNet for latent diffusion inpainting.
    Input channels:
        z_t (latent_ch) + mask (1) + known_context (latent_ch) -> 2*latent_ch + 1
    Output:
        epsilon prediction (latent_ch)
    """
    def __init__(
        self,
        latent_channels: int,
        in_channels: int,
        base_channels: int,
        channel_mults: Tuple[int, ...],
        num_res_blocks: int,
        time_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(UNetResBlock(ch, out_ch, time_dim=time_dim, dropout=dropout))
                ch = out_ch
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(ch))

        self.mid1 = UNetResBlock(ch, ch, time_dim=time_dim, dropout=dropout)
        self.mid2 = UNetResBlock(ch, ch, time_dim=time_dim, dropout=dropout)

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.ups.append(UNetResBlock(ch + out_ch, out_ch, time_dim=time_dim, dropout=dropout))
                ch = out_ch
            if i != 0:
                self.ups.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.in_conv(x)
        hs: List[torch.Tensor] = [h]

        for layer in self.downs:
            if isinstance(layer, UNetResBlock):
                h = layer(h, t_emb)
                hs.append(h)
            else:
                h = layer(h)
                hs.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for layer in self.ups:
            if isinstance(layer, UNetResBlock):
                skip = hs.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------
# Diffusion schedule
# --------------------------

@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_cumprod: torch.Tensor
    sqrt_alpha_cumprod: torch.Tensor
    sqrt_one_minus_alpha_cumprod: torch.Tensor


def make_linear_schedule(T: int, beta_start: float, beta_end: float, device=None) -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_cumprod=alpha_cumprod,
        sqrt_alpha_cumprod=torch.sqrt(alpha_cumprod),
        sqrt_one_minus_alpha_cumprod=torch.sqrt(1.0 - alpha_cumprod),
    )


# --------------------------
# LoRA modules
# --------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scale * delta


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.down = nn.Conv2d(base.in_channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        return out + self.scale * self.up(self.down(x))


def inject_lora(module: nn.Module, rank: int, alpha: float) -> None:
    """Recursively wrap all Conv2d/Linear with LoRA modules."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank=rank, alpha=alpha))
        else:
            inject_lora(child, rank, alpha)


def set_lora_enabled(module: nn.Module, enabled: bool) -> None:
    for m in module.modules():
        if isinstance(m, (LoRAConv2d, LoRALinear)):
            m.enabled = enabled


def lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for m in module.modules():
        if isinstance(m, LoRAConv2d):
            yield from m.down.parameters()
            yield from m.up.parameters()
        elif isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B


# --------------------------
# Latent Diffusion Inpainting core
# --------------------------

class LatentInpaintDiffusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.vae = ConvVAE(cfg)

        schedule = make_linear_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end)
        self.register_buffer("betas", schedule.betas)
        self.register_buffer("alphas", schedule.alphas)
        self.register_buffer("alpha_cumprod", schedule.alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", schedule.sqrt_alpha_cumprod)
        self.register_buffer("sqrt_one_minus_alpha_cumprod", schedule.sqrt_one_minus_alpha_cumprod)

        latent_ch = cfg.latent_channels
        in_ch = 2 * latent_ch + 1
        self.unet = SimpleUNet(
            latent_channels=latent_ch,
            in_channels=in_ch,
            base_channels=cfg.unet_base_channels,
            channel_mults=cfg.unet_channel_mults,
            num_res_blocks=cfg.unet_res_blocks,
            time_dim=cfg.unet_time_dim,
            dropout=cfg.unet_dropout,
        )

        inject_lora(self.unet, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
        set_lora_enabled(self.unet, enabled=False)

        self._active_lora_name: Optional[str] = None

    # --- VAE-like API ---
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    # --- Diffusion math ---
    def q_sample_inpaint(self, z0: torch.Tensor, t: torch.Tensor, mask_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(z0)
        sqrt_ac = extract(self.sqrt_alpha_cumprod, t, z0.shape)
        sqrt_om = extract(self.sqrt_one_minus_alpha_cumprod, t, z0.shape)
        z_noisy = sqrt_ac * z0 + sqrt_om * noise
        zt = z0 * (1.0 - mask_z) + z_noisy * mask_z
        return zt, noise

    def predict_x0_from_eps(self, zt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ac = extract(self.sqrt_alpha_cumprod, t, zt.shape)
        sqrt_om = extract(self.sqrt_one_minus_alpha_cumprod, t, zt.shape)
        return (zt - sqrt_om * eps) / (sqrt_ac + 1e-8)

    # --- Pipeline forward (visualization) ---
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z0 = self.encode(x)
        b, _, h, w = z0.shape
        m = downsample_mask(mask, (h, w))
        t = torch.randint(0, self.cfg.timesteps, (b,), device=z0.device).long()
        zt, _ = self.q_sample_inpaint(z0, t, m)
        z_known = z0 * (1.0 - m)
        eps_pred = self.unet(torch.cat([zt, m, z_known], dim=1), t)
        x0_pred = self.predict_x0_from_eps(zt, t, eps_pred)
        x0_pred = x0_pred * m + z0 * (1.0 - m)
        return self.decode(x0_pred)

    # --- Training ---
    def training_step(self, image: torch.Tensor, mask: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        z0 = self.encode(image)
        b, _, h, w = z0.shape
        m = downsample_mask(mask, (h, w))
        if t is None:
            t = torch.randint(0, self.cfg.timesteps, (b,), device=z0.device).long()

        zt, noise = self.q_sample_inpaint(z0, t, m)
        z_known = z0 * (1.0 - m)
        eps_pred = self.unet(torch.cat([zt, m, z_known], dim=1), t)

        mse = (eps_pred - noise) ** 2
        loss_mask = (mse * m).mean()
        loss_ctx = (mse * (1.0 - m)).mean()
        return self.cfg.mask_loss_weight * loss_mask + self.cfg.context_loss_weight * loss_ctx

    # --- Sampling (DDIM) ---
    @torch.no_grad()
    def sample(self, background_image: torch.Tensor, mask: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        z0 = self.encode(background_image)
        b, _, h, w = z0.shape
        m = downsample_mask(mask, (h, w))
        z_known = z0 * (1.0 - m)

        steps = default(num_steps, self.cfg.ddim_steps)
        T = self.cfg.timesteps
        times = torch.linspace(T - 1, 0, steps, device=z0.device).long()

        z = torch.randn_like(z0)
        z = z_known + z * m

        for i in range(steps):
            t = times[i].expand(b)
            eps = self.unet(torch.cat([z, m, z_known], dim=1), t)
            x0 = self.predict_x0_from_eps(z, t, eps)
            x0 = x0 * m + z0 * (1.0 - m)

            if i == steps - 1:
                z = x0
                break

            t_prev = times[i + 1].expand(b)
            ac_t = extract(self.alpha_cumprod, t, z.shape)
            ac_prev = extract(self.alpha_cumprod, t_prev, z.shape)

            eta = self.cfg.eta
            sigma = eta * torch.sqrt((1 - ac_prev) / (1 - ac_t)) * torch.sqrt(1 - ac_t / ac_prev)
            noise = torch.randn_like(z) if (eta > 0) else torch.zeros_like(z)

            z = torch.sqrt(ac_prev) * x0 + torch.sqrt(torch.clamp(1 - ac_prev - sigma**2, min=0.0)) * eps + sigma * noise
            z = z * m + z_known

        return self.decode(z)

    # --- LoRA control ---
    def freeze_base_unet(self) -> None:
        set_requires_grad(self.unet, False)
        for p in lora_parameters(self.unet):
            p.requires_grad = True

    def enable_lora(self, enabled: bool = True) -> None:
        set_lora_enabled(self.unet, enabled)

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = self.unet.state_dict()
        return {k: v for k, v in sd.items() if ("down.weight" in k or "up.weight" in k or "lora_A" in k or "lora_B" in k)}

    def save_lora(self, path: str) -> None:
        torch.save(self.get_lora_state_dict(), path)

    def load_lora(self, path: str, map_location: Optional[str] = "cpu") -> None:
        sd = torch.load(path, map_location=map_location)
        self.unet.load_state_dict(sd, strict=False)

    def activate_class_lora(self, class_name: str, lora_path: Optional[str] = None) -> None:
        self._active_lora_name = class_name
        self.enable_lora(True)
        if exists(lora_path):
            self.load_lora(lora_path, map_location="cpu")


# --------------------------
# Pipeline-friendly wrapper (VAE-like)
# --------------------------

class DefectPatchGenerator(nn.Module):
    """
    Plug-in module for your pipeline.

    Kept functions:
      - encode(x)
      - decode(z)
      - forward(x, mask, class_name=None)

    Added only what is necessary:
      - class_name to choose a LoRA adapter during training/sampling
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.core = LatentInpaintDiffusion(cfg)
        self.core.enable_lora(False)  # base training mode

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.core.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.core.decode(z)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, class_name: Optional[str] = None) -> torch.Tensor:
        if exists(class_name):
            self.core.activate_class_lora(class_name)
        return self.core.forward(x, mask)

    def training_step(self, image: torch.Tensor, mask: torch.Tensor, class_name: Optional[str] = None) -> torch.Tensor:
        if exists(class_name):
            self.core.activate_class_lora(class_name)
        return self.core.training_step(image=image, mask=mask)

    def freeze_base(self) -> None:
        self.core.freeze_base_unet()
        self.core.enable_lora(True)

    def start_lora_training(self, class_name: str, lora_path: Optional[str] = None) -> None:
        self.freeze_base()
        self.core.activate_class_lora(class_name, lora_path=lora_path)

    @torch.no_grad()
    def sample(self, background_image: torch.Tensor, mask: torch.Tensor, class_name: str, num_steps: Optional[int] = None) -> torch.Tensor:
        self.core.activate_class_lora(class_name)
        return self.core.sample(background_image=background_image, mask=mask, num_steps=num_steps)

    def save_class_lora(self, class_name: str, path: str) -> None:
        self.core._active_lora_name = class_name
        self.core.save_lora(path)

    def load_class_lora(self, class_name: str, path: str) -> None:
        self.core.activate_class_lora(class_name, lora_path=path)
    
    def set_active_class(self, class_name: str | None):

        self._active_class_name = class_name

    def fit_epoch(self, train_loader, val_loader, optimizer):

        device = next(self.parameters()).device
        class_name = getattr(self, "_active_class_name", None)

        # -------------------------
        # TRAIN
        # -------------------------
        self.train()
        train_total = 0.0
        n_train = 0

        for batch in train_loader:
            # image: (B,C,H,W), mask: (B,1,H,W)
            image = batch["image"].to(device)
            mask  = batch["mask"].to(device)

            loss = self.training_step(image, mask, class_name=class_name)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = image.shape[0]
            train_total += loss.item() * bs
            n_train += bs

        train_total = train_total / max(n_train, 1)

        # -------------------------
        # VAL
        # -------------------------
        self.eval()
        val_total = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                mask  = batch["mask"].to(device)

                loss = self.training_step(image, mask, class_name=class_name)

                bs = image.shape[0]
                val_total += loss.item() * bs
                n_val += bs

        val_total = val_total / max(n_val, 1)


        train_loss = {
            "total": train_total,
            "recon": train_total,
            "kl": 0.0,
            "recon_weighted": train_total,
            "kl_weighted": 0.0,
        }

        val_loss = {
            "total": val_total,
            "recon": val_total,
            "kl": 0.0,
            "recon_weighted": val_total,
            "kl_weighted": 0.0,
        }

        return train_loss, val_loss
