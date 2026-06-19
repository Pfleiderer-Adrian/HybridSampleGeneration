from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.interfaces import StepOutput
from synthesizer.mask_manipulation import TransformGenerator


class HybridVAEBase(nn.Module, ABC):
    """
    Shared implementation for VAE-style hybrid generator models.

    The generic architecture contracts live in models.interfaces. This class is
    intentionally VAE-specific: it provides padding/cropping utilities, VAE
    reparameterization, reconstruction+KL loss, training steps, checkpointing,
    and generate(mode=...) dispatch for the existing VAE/cVAE models.
    """

    cfg: Any

    @staticmethod
    def _compute_symmetric_pad(size: int, multiple: int) -> Tuple[int, int]:
        """Compute symmetric padding so that size becomes divisible by multiple."""
        if multiple <= 1:
            return (0, 0)

        remainder = size % multiple
        if remainder == 0:
            return (0, 0)

        needed = multiple - remainder
        left = needed // 2
        right = needed - left
        return left, right

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """
        Pad spatial dimensions symmetrically so each is divisible by multiple.
        Supports tensors shaped (B, C, H, W) and (B, C, D, H, W).
        """
        spatial_dims = x.ndim - 2
        if spatial_dims not in (2, 3):
            raise ValueError(
                f"Expected a 4D or 5D tensor with batch/channel axes, got shape {tuple(x.shape)}."
            )

        pad_per_dim = [
            HybridVAEBase._compute_symmetric_pad(size, multiple)
            for size in x.shape[-spatial_dims:]
        ]
        pad = tuple(value for pair in reversed(pad_per_dim) for value in pair)
        if sum(pad) == 0:
            return x, pad

        return F.pad(x, pad, mode="constant", value=0.0), pad

    @staticmethod
    def _crop_like(x: torch.Tensor, ref_shape: Tuple[int, ...]) -> torch.Tensor:
        """Center-crop the last len(ref_shape) dimensions of x to ref_shape."""
        spatial_dims = len(ref_shape)
        if spatial_dims not in (2, 3):
            raise ValueError(f"Expected a 2D or 3D reference shape, got {ref_shape!r}.")

        if x.ndim - 2 != spatial_dims:
            raise ValueError(
                f"Expected tensor shape (B, C, *ref_shape), got shape {tuple(x.shape)} "
                f"for ref_shape={ref_shape!r}."
            )

        def center_slice(current: int, target: int):
            if current == target:
                return slice(None)
            start = (current - target) // 2
            return slice(start, start + target)

        slices = [
            center_slice(current, target)
            for current, target in zip(x.shape[-spatial_dims:], ref_shape)
        ]
        return x[(..., *slices)]

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def on_epoch_start(self, epoch: int, config=None) -> None:
        """Update optional VAE KL schedule at the start of each epoch."""
        cfg = getattr(self, "cfg", None)
        if cfg is None or not hasattr(cfg, "beta_kl"):
            return

        warmup_start = getattr(cfg, "beta_kl_warmup_start", 0)
        warmup_epochs = getattr(cfg, "beta_kl_warmup_epochs", 0)
        beta_start = getattr(cfg, "beta_kl_start", 0.0)
        beta_max = getattr(cfg, "beta_kl_max", getattr(cfg, "beta_kl", 1.0))

        if warmup_start >= epoch:
            cfg.beta_kl = 0.0
            return
        if warmup_epochs <= 0:
            cfg.beta_kl = beta_max
            return

        t = min(1.0, max(0.0, epoch / warmup_epochs))
        cfg.beta_kl = beta_start + t * (beta_max - beta_start)

    def configure_optimizers(self, config):
        """Return optimizer and optional scheduler for trainable model parameters."""
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError(f"{self.__class__.__name__} has no trainable parameters.")
        return torch.optim.Adam(trainable_params, lr=config.lr), None

    def training_step(self, batch, batch_idx: int, config=None) -> StepOutput:
        """Compute one training batch loss and metrics."""
        return self._shared_step(batch)

    def validation_step(self, batch, batch_idx: int, config=None) -> StepOutput:
        """Compute one validation batch loss and metrics."""
        return self._shared_step(batch)

    def _shared_step(self, batch) -> StepOutput:
        out = self._forward_from_batch(batch)
        losses = self.loss(out)
        return StepOutput(loss=losses["total"], metrics=losses)

    def _forward_from_batch(self, batch):
        if hasattr(self, "_extract_inputs"):
            x, ori_mask, _ = self._extract_inputs(batch)
            return self(x, ori_mask, ori_mask)

        if hasattr(self, "_extract_x"):
            x = self._extract_x(batch)
            return self(x)

        if isinstance(batch, dict) and "img" in batch:
            return self(batch["img"])
        if isinstance(batch, (tuple, list)) and batch:
            return self(batch[0])
        return self(batch)

    def save_checkpoint(self, path: str, **_state) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str, **_kwargs) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def generate(self, sample, *, mode: str, **kwargs):
        """Unified generation entry point used by HybridDataGenerator."""
        mode = str(mode).lower()
        if mode in ("prior", "prior_sampling"):
            return self._generate_prior(sample, **kwargs)
        if mode in ("posterior", "posterior_sampling", "img2img"):
            return self._generate_posterior(sample, **kwargs)
        raise ValueError(f"Unknown generation mode {mode!r}. Expected 'prior' or 'posterior'.")

    def loss(self, out: dict) -> dict:
        """
        Compute the shared VAE loss.

        Expected forward output: recon, x_ref, mu, logvar.
        """
        recon = out["recon"]
        x = out["x_ref"]
        mu, logvar = out["mu"], out["logvar"]

        loss_name = str(getattr(self.cfg, "recon_loss", "smoothl1")).lower()
        if loss_name in ("smoothl1", "smooth_l1", "huber"):
            beta = float(getattr(self.cfg, "recon_smoothl1_beta", 1.0))
            try:
                recon_per_element = F.smooth_l1_loss(recon, x, reduction="none", beta=beta)
            except TypeError:
                recon_per_element = F.smooth_l1_loss(recon, x, reduction="none")
        elif loss_name in ("mse", "l2"):
            recon_per_element = (recon - x) ** 2
        else:
            raise ValueError(
                f"Unknown cfg.recon_loss={getattr(self.cfg, 'recon_loss', None)!r}. "
                "Supported: 'smoothl1' | 'mse'"
            )

        fg_weight = float(getattr(self.cfg, "fg_weight", 1.0))
        fg_threshold = float(getattr(self.cfg, "fg_threshold", 0.0))
        if fg_weight != 1.0:
            fg_mask = (x > fg_threshold).float()
            weights = torch.where(fg_mask > 0, fg_weight, 1.0)
            recon_loss = (recon_per_element * weights).mean()
        else:
            recon_loss = recon_per_element.mean()

        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl_raw = kl_per_dim.sum(dim=1).mean()

        free_bits = float(getattr(self.cfg, "free_bits", 0.0) or 0.0)
        if free_bits > 0.0:
            kl_used = kl_per_dim.clamp(min=free_bits).sum(dim=1).mean()
        else:
            kl_used = kl_raw

        recon_weighted = self.cfg.recon_weight * recon_loss
        kl_weighted = self.cfg.beta_kl * kl_used
        total = recon_weighted + kl_weighted

        return {
            "total": total,
            "recon": recon_loss,
            "kl": kl_used,
            "kl_raw": kl_raw,
            "recon_weighted": recon_weighted,
            "kl_weighted": kl_weighted,
        }

    @abstractmethod
    def warmup(self, shape, device=None, dtype=None, config=None):
        """Initialize shape-dependent or lazy layers before training/loading."""
        raise NotImplementedError

    @abstractmethod
    def _generate_posterior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        *,
        variation_strength: float = 1.0,
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def _generate_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor, None] = None,
        *,
        variation_strength: float = 1.0,
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        **kwargs,
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement prior sampling."
        )
