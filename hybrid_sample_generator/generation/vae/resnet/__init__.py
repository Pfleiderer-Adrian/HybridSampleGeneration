"""Two- and three-dimensional ResNet VAEs."""

from .model_2d import ResNetVAE2D
from .model_3d import ResNetVAE3D

__all__ = ["ResNetVAE2D", "ResNetVAE3D"]
