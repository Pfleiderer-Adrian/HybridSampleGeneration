"""Mask encoding, transformation, interpolation, and composition."""

from hybrid_sample_generator.imaging.masks.composition import combine_label_masks
from hybrid_sample_generator.imaging.masks.encoding import to_one_hot_2D, to_one_hot_3D
from hybrid_sample_generator.imaging.masks.interpolation import interpolate_masked_regions
from hybrid_sample_generator.imaging.masks.transform_generator import TransformGenerator

__all__ = [
    "TransformGenerator",
    "combine_label_masks",
    "interpolate_masked_regions",
    "to_one_hot_2D",
    "to_one_hot_3D",
]
