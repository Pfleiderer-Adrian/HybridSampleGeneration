"""Dimension-independent ResNet variational autoencoder."""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Union
import math
import numpy as np
import torch
import torch.nn as nn

from hybrid_sample_generator.generation.vae.base import HybridVAEBase
from .configuration import Config
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator


from .layers import ResNetDecoder, ResNetEncoder

# -------------------------
# VAE
# -------------------------


class ResNetVAE(HybridVAEBase):
    """
    ResNet-VAE.

    Expected input:
      - x: (B, C, H, W), float (continuous intensities; e.g. MRI)
      - No implicit clamping to [0,1]. If you want standardization/normalization,
        do it in your dataset/pipeline (recommended: z-score or robust scaling).

    Forward output:
      - recon: (B,C,H,W)
      - mu/logvar: (B,bottleneck_dim)
      - x_ref: (B,C,H,W) reference input for recon loss
    """
    def __init__(self, cfg: Config, *, in_channels: int, spatial_dims: int):
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
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
        self.spatial_dims = spatial_dims
        self.cfg = cfg
        self._validate_reconstruction_weights(cfg)
        self.in_channels = int(in_channels)

        # encoder outputs a latent feature map h
        self.encoder = ResNetEncoder(
            in_channels=self.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_multires_skips=cfg.use_multires_skips,
            spatial_dims=spatial_dims,
        )

        # decoder reconstructs from latent feature map
        self.decoder = ResNetDecoder(
            out_channels=self.in_channels,
            n_res_blocks=cfg.n_res_blocks,
            n_levels=cfg.n_levels,
            z_channels=cfg.z_channels,
            use_multires_skips=cfg.use_multires_skips,
            use_transpose_conv=cfg.use_transpose_conv,
            spatial_dims=spatial_dims,
        )

        self.fc_mu: Optional[nn.Linear] = None
        self.fc_logvar: Optional[nn.Linear] = None
        self.fc_decode: Optional[nn.Linear] = None
        self._latent_shape: Optional[Tuple[int, ...]] = None

    def _ensure_fcs(self, latent_shape: Tuple[int, ...], device: torch.device):
        """
        Lazily create (or re-create) the bottleneck fully-connected layers when latent spatial size changes.

        Inputs
        ------
        latent_shape:
            Tuple (h', w') of encoder output spatial size.
        device:
            Device to place FC layers on.

        Outputs
        -------
        None
            Side effect: initializes self.fc_mu, self.fc_logvar, self.fc_decode.
        """
        if self._latent_shape == latent_shape and self.fc_mu is not None:
            return

        self._latent_shape = latent_shape
        flat = int(self.cfg.z_channels * math.prod(latent_shape))

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
        if x.ndim != self.spatial_dims + 2:
            raise ValueError(f"Expected a batch, channel, and {self.spatial_dims} spatial dimensions, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={x.shape[1]}")

        # Continuous-valued inputs: keep intensities as-is (no auto-normalization / clamping).
        x = x.float()

        device = x.device
        B = x.shape[0]
        spatial_shape = tuple(x.shape[-self.spatial_dims:])

        # Pad spatial dims so they are divisible by 2**n_levels (required by stride-2 downsamples)
        multiple = 2 ** self.cfg.n_levels
        x_pad, pad = self._pad_to_multiple(x, multiple)

        # Encode into latent feature map
        h = self.encoder(x_pad)  # (B, z_channels, h', w')
        latent_shape = tuple(h.shape[-self.spatial_dims:])

        # Ensure FCs exist for this latent size
        self._ensure_fcs(latent_shape, device)

        # Flatten and produce mu/logvar
        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        # Sample z
        z = self.reparameterize(mu, logvar)

        # Decode: bottleneck -> latent feature map -> decoder -> recon
        h_dec = self.fc_decode(z).reshape(B, self.cfg.z_channels, *latent_shape)
        # Linear reconstruction head (no sigmoid) for continuous intensities.
        recon = self.decoder(h_dec)

        # Crop recon back to original spatial size
        recon = self._crop_like(recon, spatial_shape)
        # x_ref is the reference input used for loss (cropped/padded consistently)
        x_ref = self._crop_like(x_pad, spatial_shape) if sum(pad) else x

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
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()

        x = self._extract_x(sample)

        single = False
        if x.ndim == self.spatial_dims + 1:
            x = x.unsqueeze(0)
            single = True
        elif x.ndim == self.spatial_dims + 2:
            pass
        else:
            raise ValueError(f"Expected channel-first input with or without a batch, got {tuple(x.shape)}")

        x = x.float()

        # Continuous-valued inputs: keep intensities as-is.
        # (Optional) For legacy pipelines you can set clamp_01=True.

        if clamp_01:
            x = x.clamp(0.0, 1.0)

        x = x.to(device)

        with torch.no_grad():
            spatial_shape = tuple(x.shape[-self.spatial_dims:])
            multiple = 2 ** self.cfg.n_levels
            x_pad, _ = self._pad_to_multiple(x, multiple)

            h = model.encoder(x_pad)
            latent_shape = tuple(h.shape[-self.spatial_dims:])
            model._ensure_fcs(latent_shape, device)

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

            h_dec = model.fc_decode(z).reshape(B * n, self.cfg.z_channels, *latent_shape)
            recon = model.decoder(h_dec)
            recon = self._crop_like(recon, spatial_shape)

            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

            recon = recon.view(B, n, self.in_channels, *spatial_shape)
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
        output_shape: tuple[int, ...] | None = None,
        variation_strength: float = 1.0,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        clamp_01: bool = True,
        target_mask_generator: Optional[TransformGenerator] = None,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Generate one sample by decoding a standard-normal latent vector."""
        if variation_strength < 0:
            raise ValueError(f"variation_strength must be >= 0, got {variation_strength}")
        if output_shape is None and sample is not None:
            output_shape = tuple(self._extract_x(sample).shape[-self.spatial_dims:])
        if not isinstance(output_shape, (tuple, list)) or len(output_shape) != self.spatial_dims:
            raise ValueError(
                f"output_shape must contain {self.spatial_dims} values, got {output_shape}"
            )
        output_shape = tuple(int(value) for value in output_shape)
        if min(output_shape) <= 0:
            raise ValueError(f"output_shape must be positive, got {output_shape}")

        device = torch.device(device)
        model = self.to(device)
        model.eval()
        down = 2 ** int(self.cfg.n_levels)
        padded_shape = tuple(size + (down - size % down) % down for size in output_shape)
        latent_shape = tuple(size // down for size in padded_shape)

        with torch.no_grad():
            model._ensure_fcs(latent_shape, device)
            if variation_strength == 0.0:
                z = torch.zeros((1, int(self.cfg.bottleneck_dim)), device=device)
            else:
                z = torch.randn((1, int(self.cfg.bottleneck_dim)), device=device)
                z = z * float(variation_strength)
            decoded = model.fc_decode(z).reshape(
                1, int(self.cfg.z_channels), *latent_shape
            )
            recon = model.decoder(decoded)
            recon = self._crop_like(recon, output_shape).squeeze(0)
            if clamp_01:
                recon = recon.clamp(0.0, 1.0)

        if target_mask_generator is None:
            target_mask_generator = TransformGenerator()
        if return_torch:
            return recon, target_mask_generator.create_target_mask(
                synth_anomaly_image=recon
            )
        recon_np = recon.detach().cpu().numpy().astype(np.float32, copy=False)
        return recon_np, target_mask_generator.create_target_mask(
            synth_anomaly_image=recon_np
        )

    def warmup(self, shape, device=None, dtype=None, config=None):
        """Initialize shape-dependent fully connected layers."""
        if not isinstance(shape, (tuple, list)) or len(shape) != self.spatial_dims + 1:
            raise ValueError(
                f"shape must contain channels and {self.spatial_dims} spatial values, got {shape}"
            )
        shape = tuple(int(value) for value in shape)
        if min(shape) <= 0:
            raise ValueError(f"All dimensions must be > 0, got: {shape}")
        try:
            parameter = next(self.parameters())
            model_device, model_dtype = parameter.device, parameter.dtype
        except StopIteration:
            model_device, model_dtype = torch.device("cpu"), torch.float32
        device = model_device if device is None else torch.device(device)
        dtype = model_dtype if dtype is None else dtype
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self(torch.zeros((1, *shape), device=device, dtype=dtype))
        if was_training:
            self.train()
        return self
