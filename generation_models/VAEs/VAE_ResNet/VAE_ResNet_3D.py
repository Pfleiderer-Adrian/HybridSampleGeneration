from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable
from typing import Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from generation_models.VAEs.vae_base import HybridVAEBase
from synthesizer.mask_manipulation import TransformGenerator


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
    recon_loss:
        Reconstruction loss: "smoothl1" or "mse".
    recon_smoothl1_beta:
        Beta (delta) parameter for SmoothL1.
    use_transpose_conv:
        If True, decoder uses ConvTranspose3d for upsampling. If False, uses Upsample(trilinear)+Conv3d.
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
    beta_kl_max: float = 4.0
    beta_kl_warmup_start: int = 0
    beta_kl_warmup_epochs: int = 100
    free_bits: float = 0.0
    recon_loss: str = "smoothl1"  # 'smoothl1' or 'mse'
    recon_smoothl1_beta: float = 1.0
    use_transpose_conv: bool = True
    fg_weight: float = 1.0
    fg_threshold: float = 0.0


class ResNetVAE3D(HybridVAEBase):
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

        # 3D encoder outputs a latent feature map h
        self.encoder = ResNetEncoder3D(
            in_channels=cfg.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_multires_skips=cfg.use_multires_skips,
        )

        # 3D decoder reconstructs from latent feature map
        self.decoder = ResNetDecoder3D(
            out_channels=cfg.in_channels,
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
        x_pad, pad = self._pad_to_multiple(x, multiple)

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
        recon = self._crop_like(recon, ref_dhw)

        # x_ref is the reference input used for loss (cropped/padded consistently)
        x_ref = self._crop_like(x_pad, ref_dhw) if sum(pad) else x

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

        # If it's already a tensor
        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, np.ndarray):
            return torch.as_tensor(batch)

        # Tuple/list: assume first element is x
        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            x = batch[0]
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)

        # Dict: look for typical keys
        if isinstance(batch, dict):
            for key in ("img", "x", "image", "inputs"):
                if key in batch:
                    v = batch[key]
                    if isinstance(v, torch.Tensor):
                        return v
                    return torch.as_tensor(v)

        raise TypeError(f"Unknown batch type: {type(batch)}")

    def _generate_posterior(
            self,
            sample: Union[dict, np.ndarray, torch.Tensor],
            *,
            n: int = 1,
            variation_strength: float = 1.0,
            device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
            clamp_01: bool = True,
            target_mask_generator: Optional[TransformGenerator] = None,
            return_torch: bool = False,
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
              - (C, D, H, W)   (single sample)
        device:
            Device for inference.
        variation_strength:
            Strength of the latent variation. variation_strength=0.0 uses mu only.
        clamp_01:
            If True, clamp input and output to [0,1]. For continuous-intensity usage, set this to False.

        Outputs
        -------
        np.ndarray
            Reconstruction as float32, shape (C, D, H, W).

        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        x = self._extract_x(sample)

        x = x.float()

        # ensure batch dimension
        single = False
        if x.ndim == 4:  # (C,D,H,W) -> (1,C,D,H,W)
            x = x.unsqueeze(0)
            single = True
        elif x.ndim == 5:  # already batched (1,C,D,H,W) or (B,C,D,H,W)
            pass
        else:
            raise ValueError(f"Expected (C,D,H,W) or (B,C,D,H,W), got {tuple(x.shape)}")

        if clamp_01:
            x = x.clamp(0.0, 1.0)


        x = x.to(device)

        with torch.no_grad():
            ref_dhw = tuple(x.shape[-3:])
            multiple = 2 ** self.cfg.n_levels
            x_pad, _ = self._pad_to_multiple(x, multiple)

            h = model.encoder(x_pad)
            latent_dhw = tuple(h.shape[-3:])
            model._ensure_fcs(latent_dhw, device)

            B = x.shape[0]
            h_flat = h.reshape(B, -1)
            mu = model.fc_mu(h_flat)
            logvar = model.fc_logvar(h_flat)

            if variation_strength == 0.0:
                z = mu.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
            else:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn((B, n, mu.shape[-1]), device=device, dtype=mu.dtype)
                z = (mu.unsqueeze(1) + float(variation_strength) * std.unsqueeze(1) * eps).reshape(B * n, -1)

            h_dec = model.fc_decode(z).reshape(B * n, self.cfg.z_channels, *latent_dhw)
            recon = model.decoder(h_dec)  # (B, C, D, H, W)
            recon = self._crop_like(recon, ref_dhw)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            recon = recon.view(B, n, self.cfg.in_channels, *ref_dhw)
            if single:
                recon = recon.squeeze(0).squeeze(0)

        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()

        if return_torch:
            return recon, target_mask_generator.create_target_mask(synth_anomaly_image=recon)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, target_mask_generator.create_target_mask(synth_anomaly_image=recon_np)

    def _generate_prior(
        self,
        sample: Union[dict, np.ndarray, torch.Tensor, None] = None,
        *,
        out_dhw: tuple[int, int, int] | None = None,
        variation_strength: float = 0.5,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Generate ONE synthetic 3D sample via prior sampling.

        Samples:
            z ~ N(0, I) scaled by variation_strength, then decodes to 3D volume space.

        Parameters:
        - out_dhw: output (D, H, W). If None and sample is given, uses sample spatial size.
        - variation_strength: prior diversity strength.
        - clamp_01: clamp outputs to [0,1].
        - return_torch: return torch.Tensor instead of np.ndarray.

        Output:
        - (C, D, H, W)
        """
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

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

        down = 2 ** int(self.cfg.n_levels)
        pad_d = (down - (D % down)) % down
        pad_h = (down - (H % down)) % down
        pad_w = (down - (W % down)) % down
        D_pad, H_pad, W_pad = D + pad_d, H + pad_h, W + pad_w
        latent_dhw = (D_pad // down, H_pad // down, W_pad // down)

        z_dim = int(self.cfg.bottleneck_dim)

        with torch.no_grad():
            model._ensure_fcs(latent_dhw, device)

            if variation_strength == 0.0:
                z = torch.zeros((1, z_dim), device=device)
            else:
                z = torch.randn((1, z_dim), device=device) * float(variation_strength)

            h_dec = model.fc_decode(z).reshape(1, int(self.cfg.z_channels), *latent_dhw)
            recon = model.decoder(h_dec)
            recon = recon[..., :D, :H, :W].squeeze(0)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()

        if return_torch:
            return recon, target_mask_generator.create_target_mask(synth_anomaly_image=recon)

        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, target_mask_generator.create_target_mask(synth_anomaly_image=recon_np)

    def warmup(self, shape, device=None, dtype=None, config=None):
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
