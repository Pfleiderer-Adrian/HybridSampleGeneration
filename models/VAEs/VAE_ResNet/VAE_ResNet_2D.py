from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.model_interface import HybridModelInterface
from synthesizer.mask_manipulation import TransformGenerator


# -------------------------
# blocks (2D only)
# -------------------------
class ResidualBlock2D(nn.Module):
    """
    Basic residual block for 2D images.

    Inputs
    ------
    x:
        torch.Tensor (B, in_ch, H, W)

    Outputs
    -------
    torch.Tensor:
        (B, out_ch, H, W)
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

    Inputs
    ------
    x:
        torch.Tensor (B, C, H, W)

    Outputs
    -------
    torch.Tensor:
        Latent feature map h of shape (B, z_channels, h', w')
        (spatial dims reduced by 2**n_levels).
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

        # Initial projection to 8 channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(leak, inplace=True),
        )

        # Per-level stacks:
        # - res_stages: residual processing at current resolution
        # - down_stages: stride-2 downsampling conv
        # - skip_stages: multi-resolution skip projections to max_filters (optional)
        self.res_stages = nn.ModuleList()
        self.down_stages = nn.ModuleList()
        self.skip_stages = nn.ModuleList()

        for i in range(n_levels):
            # Stage channels grow as powers of 2
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)

            # Residual blocks at current resolution
            self.res_stages.append(
                nn.Sequential(
                    *[ResidualBlock2D(n_filters_1, n_filters_1, leak=leak) for _ in range(n_res_blocks)]
                )
            )

            # Downsample by factor 2 in each spatial axis
            self.down_stages.append(
                nn.Sequential(
                    nn.Conv2d(n_filters_1, n_filters_2, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(n_filters_2),
                    nn.LeakyReLU(leak, inplace=True),
                )
            )

            # Optional multi-resolution skip: downsample current features to a common resolution and sum
            if use_multires_skips:
                ks = 2 ** (n_levels - i)
                self.skip_stages.append(
                    nn.Sequential(
                        nn.Conv2d(n_filters_1, self.max_filters, kernel_size=ks, stride=ks, padding=0, bias=False),
                        nn.BatchNorm2d(self.max_filters),
                        nn.LeakyReLU(leak, inplace=True),
                    )
                )

        # Final projection into z_channels
        self.output_conv = nn.Conv2d(2 ** (n_levels + 3), z_channels, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            # Residual processing
            x = self.res_stages[i](x)
            if self.use_multires_skips:
                skips.append(self.skip_stages[i](x))
            # Downsample for next level
            x = self.down_stages[i](x)

        # Sum all multi-resolution skips into the deepest representation
        if self.use_multires_skips:
            x = x + torch.stack(skips, dim=0).sum(dim=0)

        return self.output_conv(x)


class ResNetDecoder2D(nn.Module):
    """
    ResNet-style 2D decoder with optional multi-resolution skip injections from the top latent.

    Inputs
    ------
    z:
        torch.Tensor (B, z_channels, h', w')

    Outputs
    -------
    torch.Tensor:
        Reconstructed feature map (B, out_channels, H, W) after upsampling.
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

        # Project latent channels to max_filters
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

        # Start from max_filters and go up in resolution
        prev_ch = self.max_filters
        for i in range(n_levels):
            # Channels shrink as we go to higher spatial resolution
            n_filters = 2 ** (self.n_levels - i + 2)

            # Upsample by factor 2
            self.up_stages.append(
                upsample_block(prev_ch, n_filters, scale=2)
            )
            prev_ch = n_filters

            # Residual refinement
            self.res_stages.append(
                nn.Sequential(
                    *[ResidualBlock2D(n_filters, n_filters, leak=leak) for _ in range(n_res_blocks)]
                )
            )

            # Optional multi-res skip injection from top feature map z_top
            if use_multires_skips:
                ks = 2 ** (i + 1)
                self.skip_stages.append(
                    upsample_block(self.max_filters, n_filters, scale=ks)
                )

        # Output reconstruction conv
        self.output_conv = nn.Conv2d(prev_ch, out_channels, 3, 1, 1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Keep a copy of the top feature map for skip injections
        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.up_stages[i](z)
            z = self.res_stages[i](z)
            if self.use_multires_skips:
                # Add injected skip feature (same shape as z) if enabled
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

    Fields
    ------
    n_res_blocks:
        Number of residual blocks per resolution level.
    n_levels:
        Number of down/up-sampling stages.
    z_channels:
        Channels in the latent feature map (before FC bottleneck).
    bottleneck_dim:
        Dimension of the VAE bottleneck vector (mu/logvar dimension).
    use_multires_skips:
        Enable multi-resolution skips in encoder/decoder.
    recon_weight:
        Weight for reconstruction loss term.
    beta_kl:
        Weight for KL divergence term.
    recon_loss:
        Reconstruction loss: "smoothl1" or "mse".
    recon_smoothl1_beta:
        Beta (delta) parameter for SmoothL1.
    use_transpose_conv:
        If True, decoder uses ConvTranspose2d for upsampling. If False, uses Upsample+Conv2d.
    fg_weight:
        Foreground weight for reconstruction loss (background weight is 1.0).
    fg_threshold:
        Foreground threshold on |x| used to build the weighting mask.
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
    beta_kl_max: float = 0.03
    beta_kl_warmup_start: int = 20
    beta_kl_warmup_epochs: int = 30
    free_bits: float = 0.0

    # --- continuous-intensity reconstruction (Option B) ---
    # Default: SmoothL1 (Huber) is typically more robust for MRI intensities than BCE.
    # Supported: "smoothl1" | "mse"
    recon_loss: str = "smoothl1"
    # Only used when recon_loss == "smoothl1". (PyTorch calls this parameter "beta".)
    recon_smoothl1_beta: float = 1.0
    # If True, decoder uses ConvTranspose2d for upsampling. If False, uses Upsample(bilinear)+Conv2d.
    use_transpose_conv: bool = True
    fg_weight: float = 1.0
    fg_threshold: float = 0.0


class ResNetVAE2D(HybridModelInterface):
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
    def __init__(self, cfg: Config):
        """
        Initialize the VAE.

        Inputs
        ------
        in_channels:
            Number of image channels (C).
        cfg:
            Config dataclass with architecture + loss weights.

        Outputs
        -------
        None
            Initializes encoder/decoder and sets up lazy FC layers (created on first forward).
        """
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels

        # 2D encoder outputs a latent feature map h
        self.encoder = ResNetEncoder2D(
            in_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_multires_skips=cfg.use_multires_skips,
        )

        # 2D decoder reconstructs from latent feature map
        self.decoder = ResNetDecoder2D(
            out_channels=cfg.in_channels,
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
        """
        Lazily create (or re-create) the bottleneck fully-connected layers when latent spatial size changes.

        Inputs
        ------
        latent_hw:
            Tuple (h', w') of encoder output spatial size.
        device:
            Device to place FC layers on.

        Outputs
        -------
        None
            Side effect: initializes self.fc_mu, self.fc_logvar, self.fc_decode.
        """
        if self._latent_hw == latent_hw and self.fc_mu is not None:
            return

        self._latent_hw = latent_hw
        flat = int(self.cfg.z_channels * math.prod(latent_hw))

        # Map latent feature map (flattened) -> bottleneck vector
        self.fc_mu = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_logvar = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        # Map bottleneck vector -> flattened latent feature map
        self.fc_decode = nn.Linear(self.cfg.bottleneck_dim, flat).to(device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder -> bottleneck -> decoder.

        Inputs
        ------
        x:
            torch.Tensor, shape (B, C, H, W)

        Outputs
        -------
        dict with:
          - recon: torch.Tensor (B,C,H,W)
          - mu: torch.Tensor (B,bottleneck_dim)
          - logvar: torch.Tensor (B,bottleneck_dim)
          - x_ref: torch.Tensor (B,C,H,W) reference input (cropped/padded)
        """
        # Validate shape
        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[1]}")

        # Continuous-valued inputs: keep intensities as-is (no auto-normalization / clamping).
        x = x.float()

        device = x.device
        B = x.shape[0]
        ref_hw = tuple(x.shape[-2:])

        # Pad spatial dims so they are divisible by 2**n_levels (required by stride-2 downsamples)
        multiple = 2 ** self.cfg.n_levels
        x_pad, pad = self._pad_to_multiple(x, multiple)

        # Encode into latent feature map
        h = self.encoder(x_pad)  # (B, z_channels, h', w')
        latent_hw = tuple(h.shape[-2:])

        # Ensure FCs exist for this latent size
        self._ensure_fcs(latent_hw, device)

        # Flatten and produce mu/logvar
        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        # Sample z
        z = self.reparameterize(mu, logvar)

        # Decode: bottleneck -> latent feature map -> decoder -> recon
        h_dec = self.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_hw)
        # Linear reconstruction head (no sigmoid) for continuous intensities.
        recon = self.decoder(h_dec)

        # Crop recon back to original spatial size
        recon = self._crop_like(recon, ref_hw)
        # x_ref is the reference input used for loss (cropped/padded consistently)
        x_ref = self._crop_like(x_pad, ref_hw) if sum(pad) else x

        return {"recon": recon, "mu": mu, "logvar": logvar, "x_ref": x_ref}

    def _extract_x(self, batch) -> torch.Tensor:
        """
        Extract the input tensor x from a batch.

        Supported batch formats
        -----------------------
        - batch is a torch.Tensor directly
        - batch is a tuple/list: (x, ...) where x is tensor-like
        - batch is a dict containing keys: 'img', 'x', 'image', or 'inputs'

        Inputs
        ------
        batch:
            Any of the supported formats.

        Outputs
        -------
        torch.Tensor
            x as a tensor (not yet moved to device).

        Raises
        ------
        TypeError
            If the batch format is unknown.
        """
        # Special case: list with exactly 2 elements and first is tensor
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
        """
        Train for one epoch over train_dataloader and (optionally) validate over val_dataloader.

        Expected dataloader output:
          - batches where x can be extracted into shape (B,C,H,W)

        Inputs
        ------
        train_dataloader:
            Iterable over training batches.
        val_dataloader:
            Iterable over validation batches (or None to skip validation).
        optimizer:
            torch optimizer instance (e.g., Adam).
        log_every:
            Update tqdm postfix every N steps.
        grad_clip_norm:
            If set, gradient clipping norm value.
        device:
            Device string or torch.device.

        Outputs
        -------
        (train_metrics, val_metrics):
            Each is a dict with averaged losses:
              - "total"
              - "recon"
              - "kl"
        """
        device = torch.device(device)
        model = self.to(device)

        def run_epoch(loader: Iterable, training: bool) -> Dict[str, float]:
            # Switch model into train/eval mode
            model.train(training)
            # Accumulators for averaged metrics
            run = {"total": 0.0, "recon": 0.0, "kl": 0.0, "kl_raw":0.0, "recon_weighted": 0.0, "kl_weighted": 0.0}
            n = 0

            # Progress bar over batches
            pbar = tqdm(loader, desc=("train" if training else "val"), leave=False, dynamic_ncols=True)
            for step, batch in enumerate(pbar, start=1):
                # Extract x from the batch in a robust way
                x = self._extract_x(batch=batch)
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)

                # Move to device
                x = x.to(device, non_blocking=True)

                # Validate shape
                if x.ndim != 4:
                    raise ValueError(f"Expected (B,C,H,W) from dataloader, got {tuple(x.shape)}")

                if training:
                    optimizer.zero_grad(set_to_none=True)

                # Enable grads only during training
                with torch.set_grad_enabled(training):
                    out = model(x)
                    losses = model.loss(out)
                    loss = losses["total"]

                    if training:
                        loss.backward()
                        # Optional gradient clipping
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()

                # Accumulate losses
                for k in run:
                    if k in losses:
                        run[k] += float(losses[k].detach().item())
                n += 1

                # Live metrics in progress bar
                if log_every and (step % log_every == 0):
                    postfix = {"total": f"{float(losses['total'].detach()):.6f}"}
                    postfix["recon"] = f"{float(losses['recon'].detach()):.6f}"
                    postfix["kl"] = f"{float(losses['kl'].detach()):.6f}"
                    pbar.set_postfix(postfix)

            # Average metrics
            denom = max(1, n)
            return {k: run[k] / denom for k in run}

        # Train epoch
        tr = run_epoch(train_dataloader, training=True)

        # Validation epoch
        va = {}
        if val_dataloader is not None:
            with torch.no_grad():
                va = run_epoch(val_dataloader, training=False)

        return tr, va

    def generate_synth_sample(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor],
        *,
        variation_strength: float = 1.0,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
    ) -> np.ndarray:
        """
        Generate a synthetic sample (reconstruction) for a SINGLE input sample.

        Internally:
          - converts the input to torch
          - ensures a batch dimension
          - samples the latent vector with z = mu + variation_strength * sigma * eps
          - returns out["recon"] as numpy

        Inputs
        ------
        sample:
            Sample dict containing the "img" artifact, or a raw tensor/array with shapes:
              - (C, H, W)   (single sample)
        device:
            Device for inference.
        variation_strength:
            Strength of the latent variation. variation_strength=0.0 uses mu only.
        clamp_01:
            If True, clamp input and output to [0,1]. For continuous-intensity usage, set this to False.

        Outputs
        -------
        np.ndarray
            Reconstruction as float32, shape (C, H, W).
        """
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        x = self._extract_x(sample)

        if x.ndim == 3:  # (C,H,W) -> (1,C,H,W)
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected (C,H,W) or (B,C,H,W), got {tuple(x.shape)}")

        x = x.float()

        # Continuous-valued inputs: keep intensities as-is.
        # (Optional) For legacy pipelines you can set clamp_01=True.

        if clamp_01:
            x = x.clamp(0.0, 1.0)

        x = x.to(device)

        with torch.no_grad():
            ref_hw = tuple(x.shape[-2:])
            multiple = 2 ** self.cfg.n_levels
            x_pad, _ = self._pad_to_multiple(x, multiple)

            h = model.encoder(x_pad)
            latent_hw = tuple(h.shape[-2:])
            model._ensure_fcs(latent_hw, device)

            B = x.shape[0]
            h_flat = h.reshape(B, -1)
            mu = model.fc_mu(h_flat)
            logvar = model.fc_logvar(h_flat)

            if variation_strength == 0.0:
                z = mu
            else:
                std = torch.exp(0.5 * logvar)
                z = mu + float(variation_strength) * std * torch.randn_like(std)

            h_dec = model.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_hw)
            recon = model.decoder(h_dec)
            recon = self._crop_like(recon, ref_hw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

        recon_np = recon.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False)
        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()
        return recon_np, target_mask_generator.create_target_mask(synth_anomaly_image=recon_np)

    def generate_synth_sample_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor, None] = None,
        *,
        out_hw: tuple[int, int] | None = None,
        variation_strength: float = 1.0,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Generate ONE synthetic sample via prior sampling.

        Samples:
            z ~ N(0, I) scaled by variation_strength, then decodes to image space.

        Parameters:
        - out_hw: output (H, W). If None and sample is given, uses sample spatial size.
        - variation_strength: prior diversity strength.
        - clamp_01: clamp outputs to [0,1].
        - return_torch: return torch.Tensor instead of np.ndarray.

        Output:
        - (C, H, W)
        """
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        if out_hw is None and sample is not None:
            out_hw = tuple(self._extract_x(sample).shape[-2:])
        if not (isinstance(out_hw, (tuple, list)) and len(out_hw) == 2):
            raise ValueError(f"out_hw must be (H,W), got {out_hw}")

        H, W = int(out_hw[0]), int(out_hw[1])
        if H <= 0 or W <= 0:
            raise ValueError(f"out_hw must be positive, got {out_hw}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        down = 2 ** int(self.cfg.n_levels)
        pad_h = (down - (H % down)) % down
        pad_w = (down - (W % down)) % down
        H_pad, W_pad = H + pad_h, W + pad_w
        latent_hw = (H_pad // down, W_pad // down)

        z_dim = int(getattr(self.cfg, "bottleneck_dim", 256))

        with torch.no_grad():
            model._ensure_fcs(latent_hw, device)

            if variation_strength == 0.0:
                z = torch.zeros((1, z_dim), device=device)
            else:
                z = torch.randn((1, z_dim), device=device) * float(variation_strength)

            h_dec = model.fc_decode(z).reshape(1, int(self.cfg.z_channels), *latent_hw)
            recon = model.decoder(h_dec)
            recon = recon[..., :H, :W].squeeze(0)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()

        if return_torch:
            return recon, target_mask_generator.create_target_mask(synth_anomaly_image=recon)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, target_mask_generator.create_target_mask(synth_anomaly_image=recon_np)

    def warmup(self, shape, device=None, dtype=None):
        """
        Warm up the model to initialize lazy FC layers (fc_mu, fc_logvar, fc_decode).

        This is necessary because FC layers depend on the latent spatial size which depends on input size.

        Inputs
        ------
        shape:
            Tuple/list (C, H, W) used to create a dummy batch (1,C,H,W).
        device:
            Optional device to run warmup on. If None, uses the model's device.
        dtype:
            Optional dtype. If None, uses model parameter dtype.

        Outputs
        -------
        self:
            Returns self for chaining.
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

        # Remember training mode, run dummy forward in eval/no_grad
        was_training = self.training
        self.eval()

        with torch.no_grad():
            x = torch.zeros((1, C, H, W), device=device, dtype=dtype)
            _ = self(x)

        # Restore mode
        if was_training:
            self.train()

        return self


if __name__ == "__main__":
    cfg = Config(n_res_blocks=4, n_levels=4, z_channels=128, bottleneck_dim=128)
    model = ResNetVAE2D(in_channels=1, cfg=cfg)
