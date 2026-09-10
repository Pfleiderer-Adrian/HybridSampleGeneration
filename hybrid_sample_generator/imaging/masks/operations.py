"""Backward-compatible exports for the former combined mask module."""

from hybrid_sample_generator.imaging.masks.encoding import to_one_hot_2D, to_one_hot_3D
from hybrid_sample_generator.imaging.masks.interpolation import interpolate_masked_regions
from hybrid_sample_generator.imaging.masks.transforms import TransformGenerator

__all__ = [
    "TransformGenerator",
    "interpolate_masked_regions",
    "to_one_hot_2D",
    "to_one_hot_3D",
]
