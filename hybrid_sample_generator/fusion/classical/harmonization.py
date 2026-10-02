"""Optional pretrained image harmonization for classical 2D fusion."""

from __future__ import annotations

from importlib import import_module

import numpy as np


SUPPORTED_HARMONIZERS = frozenset({"PCTNet", "LBM"})


class ImageHarmonizer:
    """Adapt channel-first study arrays to libcom's uint8 BGR interface."""

    def __init__(self, model_name: str, *, device=None) -> None:
        if model_name not in SUPPORTED_HARMONIZERS:
            raise ValueError(f"Unsupported image harmonization model {model_name!r}.")
        try:
            libcom = import_module("libcom")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Image harmonization requires the optional libcom dependency. "
                "Install it with `python -m pip install -e \".[harmonization]\"`."
            ) from exc

        if not hasattr(libcom, "ImageHarmonizationModel"):
            raise ImportError(
                "The installed libcom version does not provide "
                "ImageHarmonizationModel. Install the project's harmonization extra."
            )
        if device is None:
            torch = import_module("torch")
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "PCTNet and LBM harmonization through libcom require a CUDA GPU."
                )
            device = 0
        try:
            self._model = libcom.ImageHarmonizationModel(
                device=device,
                model_type=model_name,
            )
        except (AssertionError, ValueError) as exc:
            if "Only GPU are supported" in str(exc):
                raise RuntimeError(
                    "PCTNet and LBM harmonization through libcom require a CUDA GPU."
                ) from exc
            if model_name == "LBM":
                raise RuntimeError(
                    "The installed libcom build does not support the LBM harmonizer. "
                    "Install the project's harmonization extra, which pins a build "
                    "containing both PCTNet and LBM."
                ) from exc
            raise
        self.model_name = model_name

    def __call__(self, image, segmentation) -> np.ndarray:
        """Harmonize only labelled pixels and preserve the surrounding image."""
        source = np.asarray(image)
        if source.ndim != 3:
            raise ValueError(
                "Image harmonization supports only 2D channel-first images (C,H,W); "
                f"got {source.shape}."
            )
        if source.shape[0] not in (1, 3):
            raise ValueError(
                "Image harmonization supports one or three image channels; "
                f"got {source.shape[0]}."
            )
        if not np.all(np.isfinite(source)):
            raise ValueError("Image harmonization requires finite image values.")

        mask = _spatial_mask(segmentation, source.shape[1:])
        if not np.any(mask):
            return source.copy()

        encoded, minimum, value_range = _encode_uint8(source)
        if value_range == 0.0:
            return source.copy()

        rgb = (
            np.repeat(encoded, 3, axis=0)
            if encoded.shape[0] == 1
            else encoded
        )
        bgr = np.moveaxis(rgb[::-1], 0, -1)
        harmonized_bgr = np.asarray(
            self._model(bgr, mask.astype(np.uint8) * 255)
        )
        expected_shape = (*source.shape[1:], 3)
        if harmonized_bgr.shape != expected_shape:
            raise ValueError(
                f"{self.model_name} returned shape {harmonized_bgr.shape}; "
                f"expected {expected_shape}."
            )

        harmonized_rgb = np.moveaxis(harmonized_bgr, -1, 0)[::-1]
        if source.shape[0] == 1:
            # Luminance conversion prevents a generative backend from introducing
            # color into a grayscale study.
            harmonized = (
                0.299 * harmonized_rgb[0]
                + 0.587 * harmonized_rgb[1]
                + 0.114 * harmonized_rgb[2]
            )[None, ...]
        else:
            harmonized = harmonized_rgb
        decoded = minimum + harmonized.astype(np.float32) / 255.0 * value_range

        # libcom should preserve the background, but enforcing the mask here also
        # protects previous placements and the persisted ground-truth contract.
        return np.where(mask[None, ...], decoded, source).astype(
            source.dtype, copy=False
        )


def _encode_uint8(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    value_range = maximum - minimum
    if value_range == 0.0:
        return np.zeros_like(image, dtype=np.uint8), minimum, value_range
    encoded = np.rint((image - minimum) / value_range * 255.0)
    return np.clip(encoded, 0, 255).astype(np.uint8), minimum, value_range


def _spatial_mask(segmentation, spatial_shape) -> np.ndarray:
    mask = np.asarray(segmentation)
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=0)
    elif mask.ndim == 2:
        mask = mask > 0
    else:
        raise ValueError(
            "Image harmonization requires a 2D mask with optional channel axis; "
            f"got {mask.shape}."
        )
    if mask.shape != tuple(spatial_shape):
        raise ValueError(
            f"Harmonization mask shape {mask.shape} does not match image spatial "
            f"shape {tuple(spatial_shape)}."
        )
    return mask


__all__ = ["ImageHarmonizer", "SUPPORTED_HARMONIZERS"]
