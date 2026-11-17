"""Threshold foreground estimation in original image units (0..255)."""
import numpy as np
from scipy.ndimage import label, binary_fill_holes

def object_mask(image,config):
    image=np.asarray(image)
    if image.ndim not in (3,4) or image.shape[0] not in (1,3) or not np.isfinite(image).all():
        raise ValueError('Foreground expects a finite CHW/CDHW grayscale or RGB original.')
    if config.channel_reduction=='minimum':gray=image.min(axis=0)
    elif image.shape[0]==1:gray=image[0]
    else:gray=np.einsum('c,c...->...',np.array([.299,.587,.114]),image)
    mask=gray<config.threshold if config.comparison=='below' else gray>config.threshold
    labels,n=label(mask);counts=np.bincount(labels.ravel());counts[0]=0
    allowed=counts>=config.min_component_area;allowed[0]=False
    if config.components=='largest':
        candidates=np.where(allowed,counts,0)
        mask=labels==int(candidates.argmax()) if candidates.any() else np.zeros_like(mask)
    else:mask=allowed[labels]
    return binary_fill_holes(mask) if config.fill_holes else mask
