"""Conditional two- and three-dimensional ConvNeXt VAEs."""

from . import model_2d as cVAE_ConvNeXt_2D
from . import model_3d as cVAE_ConvNeXt_3D
from .model_2d import ConvNeXtcVAE2D
from .model_3d import ConvNeXtcVAE3D

__all__ = [
    "ConvNeXtcVAE2D",
    "ConvNeXtcVAE3D",
    "cVAE_ConvNeXt_2D",
    "cVAE_ConvNeXt_3D",
]
