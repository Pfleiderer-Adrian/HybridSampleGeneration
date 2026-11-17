"""Defect-aware rectangular CutMix for pixelwise supervised segmentation."""

import numpy as np
import torch


def cutmix_defect(healthy, donor, donor_mask, rng):
    """Copy one rectangle and its pixel labels; retain the healthy partner.

    Beta(1, 1) controls rectangle area. Its center is a donor defect pixel,
    guaranteeing nonempty supervision. This is a defect-aware 2D adaptation,
    not classification CutMix with area-weighted image labels.
    """
    if healthy.shape != donor.shape or donor_mask.shape != healthy[:1].shape:
        raise ValueError("CutMix image and mask shapes must align.")
    locations = torch.nonzero(donor_mask[0] > 0, as_tuple=False)
    if not len(locations):
        raise ValueError("CutMix requires a nonempty donor defect mask.")
    y, x = locations[int(rng.integers(len(locations)))].tolist()
    h, w = healthy.shape[-2:]
    ratio = np.sqrt(1 - float(rng.beta(1., 1.)))
    rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
    top, left = max(0, y - rh // 2), max(0, x - rw // 2)
    bottom, right = min(h, y + (rh + 1) // 2), min(w, x + (rw + 1) // 2)
    image, mask = healthy.clone(), torch.zeros_like(donor_mask)
    image[:, top:bottom, left:right] = donor[:, top:bottom, left:right]
    mask[:, top:bottom, left:right] = donor_mask[:, top:bottom, left:right]
    return image, mask
