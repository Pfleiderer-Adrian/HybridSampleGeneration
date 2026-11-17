"""Measurements on the final materialized binary mask."""
import numpy as np

def measure(mask,foreground=None):
    mask=np.asarray(mask)
    if mask.ndim not in (3,4):raise ValueError("Anomaly mask must have CHW/CDHW shape.")
    if not np.isfinite(mask).all():raise ValueError('Anomaly masks must be finite.')
    mask=(mask>0).any(axis=0)
    pixels=int(mask.sum());result={'mask_pixels':pixels,'mask_area_fraction':pixels/mask.size}
    if foreground is not None:
        if foreground.shape!=mask.shape:raise ValueError('Foreground and anomaly shapes differ.')
        if not foreground.any():raise ValueError('Foreground detection returned an empty object mask; check thresholds.')
        outside=int(np.count_nonzero(mask&~foreground))
        result.update(object_overlap=(pixels-outside)/pixels if pixels else 0.,outside_object_pixels=outside)
    return result
