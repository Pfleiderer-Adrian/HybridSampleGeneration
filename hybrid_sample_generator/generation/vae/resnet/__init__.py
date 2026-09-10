"""Two- and three-dimensional ResNet VAEs."""

from . import model_2d as VAE_ResNet_2D
from . import model_3d as VAE_ResNet_3D
from .model_2d import ResNetVAE2D
from .model_3d import ResNetVAE3D

__all__ = ["ResNetVAE2D", "ResNetVAE3D", "VAE_ResNet_2D", "VAE_ResNet_3D"]
