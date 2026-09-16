"""Native-resolution aligned crops and mask-independent tiled inference."""

from itertools import islice, product
import numpy as np
import torch
from torch.nn import functional as F


def crop_training_patch(image, target, mask, size, rng, *, anomalous=False):
    """Crop aligned CHW arrays; anomalous crops include a randomly selected defect pixel."""
    height, width = image.shape[-2:]
    ph, pw = size
    if target.shape != image.shape or mask.shape[-2:] != (height, width):
        raise ValueError("Patch image, target and mask must be aligned.")
    if anomalous:
        foreground = np.flatnonzero((mask > 0).any(axis=0))
        if not len(foreground):
            raise ValueError("Cannot sample an anomalous patch from an empty mask.")
        y, x = divmod(int(rng.choice(foreground)), width)
        top = int(rng.integers(max(0, y-ph+1), min(y, max(0, height-ph))+1))
        left = int(rng.integers(max(0, x-pw+1), min(x, max(0, width-pw))+1))
    else:
        top = int(rng.integers(max(0, height-ph)+1))
        left = int(rng.integers(max(0, width-pw)+1))
    slices = (slice(None), slice(top, top+ph), slice(left, left+pw))
    result = []
    for array, mode in ((image, "edge"), (target, "edge"), (mask, "constant")):
        crop = array[slices]
        padding = ((0, 0), (0, ph-crop.shape[-2]), (0, pw-crop.shape[-1]))
        result.append(np.pad(crop, padding, mode=mode))
    return tuple(result)


def tile_starts(length, patch, overlap):
    if length <= patch:
        return [0]
    stride = max(1, int(patch * (1-overlap)))
    starts = list(range(0, length-patch+1, stride))
    if starts[-1] != length-patch:
        starts.append(length-patch)
    return starts


@torch.no_grad()
def tiled_logits(model, image, device, size, overlap, batch_size):
    """Infer one CHW image using bounded GPU batches; average logits on CPU.

    No ground-truth mask is accepted. All original pixels are covered; padding
    is discarded before scoring. Only patches, never full images, go to the GPU.
    """
    if batch_size < 1 or not 0 <= overlap < 1:
        raise ValueError("Invalid inference batch size or overlap.")
    image = image.cpu()
    height, width = image.shape[-2:]
    ph, pw = size
    padded = F.pad(image[None], (0, max(0, pw-width), 0, max(0, ph-height)), mode="replicate")[0]
    h, w = padded.shape[-2:]
    summed = torch.zeros((2, h, w), dtype=torch.float32)
    counts = torch.zeros((1, h, w), dtype=torch.float32)
    positions = iter(product(tile_starts(h, ph, overlap), tile_starts(w, pw, overlap)))
    while batch := list(islice(positions, batch_size)):
        patches = torch.stack([padded[:, y:y+ph, x:x+pw] for y, x in batch]).to(device)
        _, logits = model(patches)
        logits = logits.float().cpu()
        for (y, x), prediction in zip(batch, logits):
            summed[:, y:y+ph, x:x+pw] += prediction
            counts[:, y:y+ph, x:x+pw] += 1
    return (summed / counts)[:, :height, :width]
