"""Conditional two- and three-dimensional ConvNeXt VAEs."""

from .model_2d import ConvNeXtcVAE2D
from .model_3d import ConvNeXtcVAE3D

__all__ = ["ConvNeXtcVAE2D", "ConvNeXtcVAE3D"]
