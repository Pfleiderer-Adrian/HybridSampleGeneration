"""Two- and three-dimensional ConvNeXt VAEs."""

from . import model_2d as VAE_ConvNeXt_2D
from . import model_3d as VAE_ConvNeXt_3D
from .model_2d import ConvNeXtVAE2D
from .model_3d import ConvNeXtVAE3D

__all__ = ["ConvNeXtVAE2D", "ConvNeXtVAE3D", "VAE_ConvNeXt_2D", "VAE_ConvNeXt_3D"]
