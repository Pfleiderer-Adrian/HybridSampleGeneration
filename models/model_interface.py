from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class HybridModelInterface(nn.Module, ABC):
    """
    Abstract interface for models used by Trainer and HybridDataGenerator.

    This class documents the external contract and hosts behavior shared by all
    current generator models.
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
            HybridModelInterface._compute_symmetric_pad(size, multiple)
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
        """
        Reparameterization trick: sample z ~ N(mu, sigma^2) using mu + eps*sigma.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(self, out: dict) -> dict:
        """
        Compute the shared VAE loss for all hybrid generator models.

        Expected forward output:
            recon, x_ref, mu, logvar

        Returned metrics are intentionally stable because synthesizer.Trainer
        logs these keys for every model.
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
    def warmup(self, shape, device=None, dtype=None):
        """
        Initialize shape-dependent or lazy layers before optimizer creation or checkpoint loading.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_epoch(
        self,
        train_dataloader,
        val_dataloader,
        optimizer,
        *,
        log_every=1,
        grad_clip_norm: Optional[float] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Tuple[dict, dict]:
        """
        Train one epoch and optionally validate.

        Expected by synthesizer.Trainer.train().
        Implementations should return train_metrics and val_metrics. Both dicts
        should include at least "total", "recon", "kl", "recon_weighted",
        "kl_weighted", and "kl_raw" when validation is used by the trainer.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_synth_sample(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        *,
        clamp_01: bool = True,
        **kwargs,
    ):
        """
        Generate a synthetic anomaly sample from an existing anomaly sample.

        Expected by synthesizer.HybridDataGenerator.generate_synth_anomalies().
        Implementations should return synthetic_image and target_mask.
        """
        raise NotImplementedError

    def generate_synth_sample_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor, None] = None,
        *,
        clamp_01: bool = True,
        **kwargs,
    ):
        """
        Optional prior-sampling entry point used when config.prior_sampling is true.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement prior sampling."
        )

    @staticmethod
    def _create_tgt_mask_from_synth_anomaly(
        synth_anomaly_image: Union[np.ndarray, torch.Tensor],
        background_threshold: Optional[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        if torch.is_tensor(synth_anomaly_image):
            if background_threshold is None:
                threshold = torch.nanmin(synth_anomaly_image) + 0.001
            else:
                threshold = torch.as_tensor(
                    background_threshold,
                    dtype=synth_anomaly_image.dtype,
                    device=synth_anomaly_image.device,
                )
            synth_projection = torch.amax(synth_anomaly_image, dim=0)
            return (synth_projection > threshold).to(torch.uint8)

        if background_threshold is None:
            threshold = float(np.nanmin(synth_anomaly_image)) + 0.001
        else:
            threshold = float(background_threshold)
        synth_projection = np.max(synth_anomaly_image, axis=0)
        return (synth_projection > threshold).astype(np.uint8)
