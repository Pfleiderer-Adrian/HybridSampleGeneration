from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable
from typing import Union
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
    """
    Compute symmetric (left,right) padding so that `size` becomes divisible by `multiple`.

    Inputs
    ------
    size:
        Current length of one spatial axis (e.g., D or H or W).
    multiple:
        The required divisibility constraint (e.g., 2**n_levels).

    Outputs
    -------
    (left_pad, right_pad):
        Tuple[int,int] indicating how many zeros to pad on each side.
    """
    if multiple <= 1:
        return (0, 0)

    # Remainder when dividing by 'multiple'
    r = size % multiple
    if r == 0:
        return (0, 0)

    # We need to add 'need' voxels to reach the next divisible size
    need = multiple - r
    left = need // 2
    right = need - left
    return left, right


def _pad_to_multiple_3d(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Pad a 5D tensor (B, C, D, H, W) so that D/H/W are divisible by `multiple`.

    PyTorch F.pad order for 5D is: (wL, wR, hL, hR, dL, dR)

    Inputs
    ------
    x:
        torch.Tensor of shape (B, C, D, H, W).
    multiple:
        Integer constraint for D/H/W divisibility.

    Outputs
    -------
    x_padded:
        torch.Tensor padded with zeros to meet divisibility.
    pad:
        Tuple[int,...] length 6: (wL,wR,hL,hR,dL,dR)
        Useful to decide whether cropping back is needed.
    """
    # Extract spatial sizes
    d, h, w = x.shape[-3:]

    # Compute symmetric padding per axis
    dL, dR = _compute_symmetric_pad(d, multiple)
    hL, hR = _compute_symmetric_pad(h, multiple)
    wL, wR = _compute_symmetric_pad(w, multiple)

    # F.pad expects (wL,wR,hL,hR,dL,dR)
    pad = (wL, wR, hL, hR, dL, dR)

    # If no padding needed, return as-is
    if sum(pad) == 0:
        return x, pad

    # Pad with constant zeros (background)
    return F.pad(x, pad, mode="constant", value=0.0), pad


def _crop_like_3d(x: torch.Tensor, ref_dhw: Tuple[int, int, int]) -> torch.Tensor:
    """
    Center-crop a tensor spatially to match a reference size (D,H,W).

    Inputs
    ------
    x:
        torch.Tensor of shape (..., D, H, W)
    ref_dhw:
        Tuple (D_ref, H_ref, W_ref) to crop to.

    Outputs
    -------
    torch.Tensor:
        Cropped tensor with spatial shape exactly ref_dhw.
    """
    d_ref, h_ref, w_ref = ref_dhw
    d, h, w = x.shape[-3:]

    # Helper: build a slice that center-crops from `cur` to `ref`
    def sl(cur: int, ref: int):
        if cur == ref:
            return slice(None)
        start = (cur - ref) // 2
        return slice(start, start + ref)

    # Apply slices on the last 3 axes
    return x[..., sl(d, d_ref), sl(h, h_ref), sl(w, w_ref)]


# -------------------------
# blocks (3D only)
# -------------------------
class ResidualBlock3D(nn.Module):
    """
    Basic residual block for 3D volumes.

    Inputs
    ------
    x:
        torch.Tensor (B, in_ch, D, H, W)

    Outputs
    -------
    torch.Tensor:
        (B, out_ch, D, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int, leak: float = 0.2):
        super().__init__()
        # First conv + BN + LeakyReLU
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act1 = nn.LeakyReLU(leak, inplace=True)

        # Second conv + BN + LeakyReLU
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act2 = nn.LeakyReLU(leak, inplace=True)

        # Projection for residual connection if channel count changes
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv3d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity path (optionally projected)
        identity = x if self.proj is None else self.proj(x)

        # Residual path
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))

        # Residual addition
        return identity + out


class ResNetEncoder3D(nn.Module):
    """
    ResNet-style 3D encoder with multi-resolution skip aggregation.

    Inputs
    ------
    x:
        torch.Tensor (B, C, D, H, W)

    Outputs
    -------
    torch.Tensor:
        Latent feature map h of shape (B, z_channels, d', h', w')
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

        # max_filters is used as a shared target channels for multi-res skip paths
        self.max_filters = 2 ** (n_levels + 3)

        # Initial projection to 8 channels
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_channels, 8, 3, 1, 1, bias=False),
            nn.BatchNorm3d(8),
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
                    *[
                        ResidualBlock3D(n_filters_1, n_filters_1, leak=leak)
                        for _ in range(n_res_blocks)
                    ]
                )
            )

            # Downsample by factor 2 in each spatial axis
            self.down_stages.append(
                nn.Sequential(
                    nn.Conv3d(n_filters_1, n_filters_2, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.BatchNorm3d(n_filters_2),
                    nn.LeakyReLU(leak, inplace=True),
                )
            )

            # Optional multi-resolution skip: downsample current features to a common resolution and sum
            if use_multires_skips:
                ks = 2 ** (n_levels - i)  # kernel/stride chosen so all skip outputs align
                self.skip_stages.append(
                    nn.Sequential(
                        nn.Conv3d(n_filters_1, self.max_filters, kernel_size=ks, stride=ks, padding=0, bias=False),
                        nn.BatchNorm3d(self.max_filters),
                        nn.LeakyReLU(leak, inplace=True),
                    )
                )

        # Final projection into z_channels
        self.output_conv = nn.Conv3d(2 ** (n_levels + 3), z_channels, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            # Residual processing
            x = self.res_stages[i](x)

            # Collect skip projection if enabled
            if self.use_multires_skips:
                skips.append(self.skip_stages[i](x))

            # Downsample for next level
            x = self.down_stages[i](x)

        # Sum all multi-resolution skips into the deepest representation
        if self.use_multires_skips:
            x = x + torch.stack(skips, dim=0).sum(dim=0)

        return self.output_conv(x)


class ResNetDecoder3D(nn.Module):
    """
    ResNet-style 3D decoder with multi-resolution skip injections from the top latent.

    Inputs
    ------
    z:
        torch.Tensor (B, z_channels, d', h', w')

    Outputs
    -------
    torch.Tensor:
        Reconstructed feature map (B, out_channels, D, H, W) after upsampling.
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
            nn.Conv3d(z_channels, self.max_filters, 3, 1, 1, bias=False),
            nn.BatchNorm3d(self.max_filters),
            nn.LeakyReLU(leak, inplace=True),
        )

        self.up_stages = nn.ModuleList()
        self.res_stages = nn.ModuleList()
        self.skip_stages = nn.ModuleList()

        def upsample_block(in_ch: int, out_ch: int, scale: int) -> nn.Sequential:
            if self.use_transpose_conv:
                return nn.Sequential(
                    nn.ConvTranspose3d(in_ch, out_ch, kernel_size=scale, stride=scale, padding=0, bias=False),
                    nn.BatchNorm3d(out_ch),
                    nn.LeakyReLU(leak, inplace=True),
                )
            return nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=False),
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
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
                    *[
                        ResidualBlock3D(n_filters, n_filters, leak=leak)
                        for _ in range(n_res_blocks)
                    ]
                )
            )

            # Optional multi-res skip injection from top feature map z_top
            if use_multires_skips:
                ks = 2 ** (i + 1)
                self.skip_stages.append(
                    upsample_block(self.max_filters, n_filters, scale=ks)
                )

        # Output reconstruction conv
        self.output_conv = nn.Conv3d(prev_ch, out_channels, 3, 1, 1, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Keep a copy of the top feature map for skip injections
        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.up_stages[i](z)
            z = self.res_stages[i](z)

            # Add injected skip feature (same shape as z) if enabled
            if self.use_multires_skips:
                z = z + self.skip_stages[i](z_top)

        return self.output_conv(z)


# -------------------------
# VAE 3D
# -------------------------
@dataclass
class Config:
    """
    Hyperparameters for ResNetVAE3D.

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
    use_transpose_conv:
        If True, decoder uses ConvTranspose3d for upsampling. If False, uses Upsample(trilinear)+Conv3d.
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


class ResNetVAE3D(nn.Module):
    """
    3D ResNet-VAE.

    Expected input:
      - x: (B, C, D, H, W), float (continuous intensities).

    Notes
    -----
    This VAE variant uses a *continuous* reconstruction objective (MSE / SmoothL1) and a
    *linear* decoder output (no sigmoid). Therefore we do **not** clamp or auto-rescale inputs
    inside the model. Do any desired normalization in your dataset / pipeline (e.g., per-scan
    z-score, robust median/MAD, or percentile clipping).

    Forward output is a dict:
      - recon: reconstructed x (B,C,D,H,W)
      - mu: mean vector (B, bottleneck_dim)
      - logvar: log-variance vector (B, bottleneck_dim)
      - x_ref: reference input used for reconstruction loss (cropped/padded version)
    """
    def __init__(self, in_channels: int, cfg: Config):
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
        self.in_channels = in_channels

        # 3D encoder outputs a latent feature map h
        self.encoder = ResNetEncoder3D(
            in_channels=in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_multires_skips=cfg.use_multires_skips,
        )

        # 3D decoder reconstructs from latent feature map
        self.decoder = ResNetDecoder3D(
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
        self._latent_dhw: Optional[Tuple[int, int, int]] = None

    def _ensure_fcs(self, latent_dhw: Tuple[int, int, int], device: torch.device):
        """
        Lazily create (or re-create) the bottleneck fully-connected layers when latent spatial size changes.

        Inputs
        ------
        latent_dhw:
            Tuple (d', h', w') of encoder output spatial size.
        device:
            Device to place FC layers on.

        Outputs
        -------
        None
            Side effect: initializes self.fc_mu, self.fc_logvar, self.fc_decode.
        """
        # If FCs already match this latent shape, do nothing
        if self._latent_dhw == latent_dhw and self.fc_mu is not None:
            return

        self._latent_dhw = latent_dhw
        flat = int(self.cfg.z_channels * math.prod(latent_dhw))

        # Map latent feature map (flattened) -> bottleneck vector
        self.fc_mu = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)
        self.fc_logvar = nn.Linear(flat, self.cfg.bottleneck_dim).to(device)

        # Map bottleneck vector -> flattened latent feature map
        self.fc_decode = nn.Linear(self.cfg.bottleneck_dim, flat).to(device)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample z ~ N(mu, sigma^2) using mu + eps*sigma.

        Inputs
        ------
        mu:
            torch.Tensor (B, bottleneck_dim)
        logvar:
            torch.Tensor (B, bottleneck_dim)

        Outputs
        -------
        torch.Tensor
            z: (B, bottleneck_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder -> bottleneck -> decoder.

        Inputs
        ------
        x:
            torch.Tensor, shape (B, C, D, H, W)

        Outputs
        -------
        dict with:
          - recon: torch.Tensor (B,C,D,H,W)
          - mu: torch.Tensor (B,bottleneck_dim)
          - logvar: torch.Tensor (B,bottleneck_dim)
          - x_ref: torch.Tensor (B,C,D,H,W) reference input (cropped/padded)
        """
        # Validate shape
        if x.ndim != 5:
            raise ValueError(f"Expected (B,C,D,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[1]}")

        x = x.float()

        # NOTE: Continuous-intensity model: do NOT auto-rescale or clamp here.
        # Apply any normalization outside the model (recommended).

        device = x.device
        B = x.shape[0]
        ref_dhw = tuple(x.shape[-3:])  # original spatial size

        # Pad spatial dims so they are divisible by 2**n_levels (required by stride-2 downsamples)
        multiple = 2 ** self.cfg.n_levels
        x_pad, pad = _pad_to_multiple_3d(x, multiple)

        # Encode into latent feature map
        h = self.encoder(x_pad)  # (B, z_channels, d', h', w')
        latent_dhw = tuple(h.shape[-3:])

        # Ensure FCs exist for this latent size
        self._ensure_fcs(latent_dhw, device)

        # Flatten and produce mu/logvar
        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        # Sample z
        z = self.reparameterize(mu, logvar)

        # Decode: bottleneck -> latent feature map -> decoder -> recon
        h_dec = self.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_dhw)
        recon = self.decoder(h_dec)  # linear output for continuous intensities

        # Crop recon back to original spatial size
        recon = _crop_like_3d(recon, ref_dhw)

        # x_ref is the reference input used for loss (cropped/padded consistently)
        x_ref = _crop_like_3d(x_pad, ref_dhw) if sum(pad) else x

        return {"recon": recon, "mu": mu, "logvar": logvar, "x_ref": x_ref}


    def loss(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss = recon_weight * ReconLoss(recon,x) + beta_kl * KL(mu,logvar).

        Inputs
        ------
        out:
            dict from forward() containing:
              - recon, x_ref, mu, logvar

        Outputs
        -------
        dict:
          - total: torch.Tensor scalar
          - recon: torch.Tensor scalar (BCE)
          - kl: torch.Tensor scalar
        """
        recon = out["recon"]
        x = out["x_ref"]
        mu, logvar = out["mu"], out["logvar"]

        # Reconstruction loss (mean over all voxels)
        # Reconstruction loss (mean over all voxels)
        if getattr(self.cfg, "recon_loss", "smoothl1").lower() == "mse":
            recon_loss = F.mse_loss(recon, x, reduction="mean")
        else:
            # SmoothL1 (Huber) is usually a good default for medical volumes (robust to outliers).
            beta = float(getattr(self.cfg, "recon_smoothl1_beta", 1.0))
            try:
                recon_loss = F.smooth_l1_loss(recon, x, reduction="mean", beta=beta)
            except TypeError:
                # Older PyTorch without 'beta' argument
                recon_loss = F.smooth_l1_loss(recon, x, reduction="mean")


        # KL divergence term (mean over batch)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        recon_weighted = self.cfg.recon_weight * recon_loss
        kl_weighted = self.cfg.beta_kl * kl

        total = recon_weighted + kl_weighted
        return {"total": total, "recon": recon_loss, "kl": kl, "recon_weighted": recon_weighted, "kl_weighted": kl_weighted}

    def _extract_x(self, batch) -> torch.Tensor:
        """
        Extract the input tensor x from a batch.

        Supported batch formats
        -----------------------
        - batch is a torch.Tensor directly
        - batch is a tuple/list: (x, ...) where x is tensor-like
        - batch is a dict containing keys: 'x', 'image', or 'inputs'

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

        # If it's already a tensor
        if isinstance(batch, torch.Tensor):
            return batch

        # Tuple/list: assume first element is x
        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            x = batch[0]
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)

        # Dict: look for typical keys
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
        """
        Train for one epoch over train_dataloader and (optionally) validate over val_dataloader.

        Expected dataloader output:
          - batches where x can be extracted into shape (B,C,D,H,W)

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
            run = {"total": 0.0, "recon": 0.0, "kl": 0.0, "recon_weighted":0.0, "kl_weighted":0.0}
            n = 0

            # Progress bar over batches
            pbar = tqdm(loader, desc=("train" if training else "val"), leave=False, dynamic_ncols=True)

            for step, batch in enumerate(pbar, start=1):
                # Extract x from the batch in a robust way
                x = self._extract_x(batch=batch)

                # Ensure tensor type
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)

                # Move to device
                x = x.to(device, non_blocking=True)

                # Validate shape
                if x.ndim != 5:
                    raise ValueError(f"Expected (B,C,D,H,W) from dataloader, got {tuple(x.shape)}")

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
                    if "recon" in losses:
                        postfix["recon"] = f"{float(losses['recon'].detach()):.6f}"
                    if "kl" in losses:
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
            sample: np.ndarray,
            *,
            device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
            clamp_01: bool = True,
    ) -> np.ndarray:
        """
        Generate a synthetic sample (reconstruction) for a SINGLE input sample.

        Internally:
          - converts the input to torch
          - ensures a batch dimension
          - runs forward()
          - returns out["recon"] as numpy

        Inputs
        ------
        sample:
            np.ndarray expected shapes:
              - (C, D, H, W)   (single sample)
            (If you pass tensors or include batch dims, behavior is ambiguous in current code.)
        device:
            Device for inference.
        clamp_01:
            If True, clamp input and output to [0,1]. For continuous-intensity usage, set this to False.

        Outputs
        -------
        np.ndarray
            Reconstruction as float32, shape (C, D, H, W).

        """
        device = torch.device(device)
        model = self.to(device)
        model.eval()

        # Convert numpy -> torch
        if not isinstance(sample, torch.Tensor):
            x = torch.as_tensor(sample)
        else:
            x = sample

        x = x.float()

        # ensure batch dimension
        if x.ndim == 4:  # (C,D,H,W) -> (1,C,D,H,W)
            x = x.unsqueeze(0)
        elif x.ndim == 5:  # already batched (1,C,D,H,W) or (B,C,D,H,W)
            pass
        else:
            raise ValueError(f"Expected (C,D,H,W) or (B,C,D,H,W), got {tuple(x.shape)}")

        if clamp_01:
            x = x.clamp(0.0, 1.0)


        x = x.to(device)

        with torch.no_grad():
            out = model(x)
            recon = out["recon"]  # (B, C, D, H, W)
            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

        recon = recon.detach().cpu()

        # Remove batch dimension: (1,C,...) -> (C,...)
        recon_np = recon.squeeze(0).numpy().astype(np.float32, copy=False)
        return recon_np

    def warmup(self, shape, device=None, dtype=None):
        """
        Warm up the model to initialize lazy FC layers (fc_mu, fc_logvar, fc_decode).

        This is necessary because FC layers depend on the latent spatial size which depends on input size.

        Inputs
        ------
        shape:
            Tuple/list (C, D, H, W) used to create a dummy batch (1,C,D,H,W).
        device:
            Optional device to run warmup on. If None, uses the model's device.
        dtype:
            Optional dtype. If None, uses model parameter dtype.

        Outputs
        -------
        self:
            Returns self for chaining.
        """
        # Validate shape format
        if not (isinstance(shape, (tuple, list)) and len(shape) == 4):
            raise ValueError(f"shape must be (C,D,H,W), got: {shape}")

        C, D, H, W = map(int, shape)
        if min(C, D, H, W) <= 0:
            raise ValueError(f"All dimensions must be > 0, got: {shape}")

        # Infer device/dtype from parameters if possible
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
            x = torch.zeros((1, C, D, H, W), device=device, dtype=dtype)
            _ = self(x)  # triggers _ensure_fcs(...) inside forward

        # Restore mode
        if was_training:
            self.train()

        return self

if __name__ == "__main__":
    # debug
    cfg = Config(n_res_blocks=4, n_levels=4, z_channels=128, bottleneck_dim=128)
    model = ResNetVAE3D(in_channels=1, cfg=cfg)
