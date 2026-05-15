import numpy as np
import torch
import torch.nn.functional as F

def augment_mask(mask: np.ndarray):
    return mask

def to_one_hot(mask: torch.Tensor, mask_channels: int) -> torch.Tensor:
    """Converts 3D/4D integer masks to 5D one-hot float tensors of shape (B, C, D, H, W)."""
    
    # already 5D and one hot encoded (more than one Channel)
    if mask.ndim == 5 and mask.shape[1] > 1:
        return mask.float()
        
    # 4D: (C, D, H, W) with C > 1 (only batch missing?)
    if mask.ndim == 4 and mask.shape[0] > 1:
        return mask.unsqueeze(0).float()

    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask.squeeze(1) # (B, 1, D, H, W) -> (B, D, H, W)
    elif mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0) # (1, D, H, W) -> (D, H, W)
        
    mask = mask.long()
    
    # add batch dimension if necessary
    if mask.ndim == 3:
        mask = mask.unsqueeze(0) # -> (1, D, H, W)
        
    # need (B, D, H, W) before one-hot encoding
    if mask.ndim != 4:
        raise ValueError(f"Erwartete Masken-Shape (B, D, H, W) oder (D, H, W), bekommen: {mask.shape}")
        
    num_classes = mask_channels + 1    # add background class
    mask_oh = F.one_hot(mask, num_classes=num_classes) # (B, D, H, W, C)
    
    mask_oh = mask_oh[..., 1:]  # remove background channel (class 0)
        
    return mask_oh.permute(0, 4, 1, 2, 3).float()   # (B, D, H, W, C) -> (B, C, D, H, W)