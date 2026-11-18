"""Window geometry of the 2x AnomalyDiffusion profile (NumPy/PIL only).

Defects are processed in native 128x128 windows centered on each mask
component (64 px stride for components larger than 128 px). Windows are
upscaled 2x to the 256x256 resolution of the latent diffusion model and the
generated content is scaled back before it is written into the native image.
"""
import numpy as np
from PIL import Image

WINDOW = 128
SCALE = 2


def centers(lo, hi, window=WINDOW):
    """Window centers along one axis that cover the interval ``[lo, hi)``."""
    if hi - lo <= window:
        return [(lo + hi) // 2]
    half = window // 2
    result = list(range(lo + half, hi - half + 1, half))
    if result[-1] != hi - half:
        result.append(hi - half)
    return result


def centered(array, cx, cy, size):
    """Crop ``size`` x ``size`` around a fixed center; zero-pad outside the image."""
    x, y = cx - size // 2, cy - size // 2
    out = np.zeros((size, size) + array.shape[2:], dtype=array.dtype)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(array.shape[1], x + size), min(array.shape[0], y + size)
    out[y0 - y:y1 - y, x0 - x:x1 - x] = array[y0:y1, x0:x1]
    return out, [x, y, size, size]


def upscale_image(image):
    return np.asarray(Image.fromarray(image).resize((WINDOW * SCALE,) * 2, Image.Resampling.BICUBIC))


def upscale_mask(mask):
    return np.asarray(Image.fromarray(mask.astype("uint8") * 255).resize((WINDOW * SCALE,) * 2, Image.Resampling.NEAREST)) > 0


def defect_windows(image, mask):
    """Training windows covering every defect pixel of a full-resolution image.

    Yields ``(window_image_2x, window_mask_2x, crop_xywh, component)``.
    """
    from scipy.ndimage import find_objects, label

    labels, _ = label(mask)
    seen, coverage = set(), np.zeros_like(mask, dtype=bool)
    for component, region in enumerate(find_objects(labels), start=1):
        for cy in centers(region[0].start, region[0].stop):
            for cx in centers(region[1].start, region[1].stop):
                support, _ = centered(labels == component, cx, cy, WINDOW)
                if not support.any() or (cx, cy) in seen:
                    continue
                seen.add((cx, cy))
                native_mask, crop = centered(mask, cx, cy, WINDOW)
                native_image, _ = centered(image, cx, cy, WINDOW)
                scaled_mask = upscale_mask(native_mask)
                if int(scaled_mask.sum()) != int(native_mask.sum()) * SCALE ** 2:
                    raise ValueError("Mask scaling lost foreground.")
                x, y, _, _ = crop
                coverage[max(y, 0):min(y + WINDOW, mask.shape[0]), max(x, 0):min(x + WINDOW, mask.shape[1])] = True
                yield upscale_image(native_image), scaled_mask, crop, component
    if np.any(mask & ~coverage):
        raise ValueError("Defect windows do not cover every mask pixel.")


def windows(mask):
    """Native 128 windows inside a 256 crop that cover every foreground pixel."""
    from scipy.ndimage import find_objects, label

    labels, _ = label(mask)
    coords = set()
    for component, region in enumerate(find_objects(labels), 1):
        for cy in centers(region[0].start, region[0].stop):
            for cx in centers(region[1].start, region[1].stop):
                x, y = min(WINDOW, max(0, cx - WINDOW // 2)), min(WINDOW, max(0, cy - WINDOW // 2))
                if (labels[y:y + WINDOW, x:x + WINDOW] == component).any():
                    coords.add((x, y))
    cover = np.zeros(mask.shape, bool)
    for x, y in coords:
        cover[y:y + WINDOW, x:x + WINDOW] = True
    if not mask.any() or np.any(mask & ~cover):
        raise ValueError("Empty mask or incomplete native foreground coverage.")
    return sorted(coords)


def stitch(healthy, generated):
    """Scale decoded windows back, average overlaps and keep the outer context.

    No mask-based compositing: background changes inside each generated window
    remain part of the AnomalyDiffusion output.
    """
    total = np.zeros(healthy.shape, np.float64)
    weight = np.zeros(healthy.shape[:2], np.uint16)
    for x, y, value in generated:
        native = np.asarray(Image.fromarray(value).resize((WINDOW, WINDOW), Image.Resampling.BICUBIC))
        total[y:y + WINDOW, x:x + WINDOW] += native
        weight[y:y + WINDOW, x:x + WINDOW] += 1
    out = healthy.copy()
    active = weight > 0
    out[active] = np.clip(np.rint(total[active] / weight[active, None]), 0, 255).astype("uint8")
    return out


def crop_around_component(image, mask, rng, size=2 * WINDOW):
    """Pick one mask component at random and crop ``size`` x ``size`` around it.

    ``image`` is HWC uint8 RGB and ``mask`` a 2D boolean array of the same size.
    Returns ``(healthy_crop, mask_crop, crop_xywh, component)``.
    """
    from scipy.ndimage import find_objects, label

    labels, n = label(mask)
    if not n:
        raise ValueError("Empty hybrid mask.")
    component = int(rng.integers(1, n + 1))
    region = find_objects(labels)[component - 1]
    cy, cx = (region[0].start + region[0].stop) // 2, (region[1].start + region[1].stop) // 2
    # Keep a positive center even for hollow or non-convex large components.
    if not (labels[max(0, cy - WINDOW):cy + WINDOW, max(0, cx - WINDOW):cx + WINDOW] == component).any():
        ys, xs = np.nonzero(labels == component)
        j = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        cy, cx = int(ys[j]), int(xs[j])
    mask_crop, crop = centered(mask, cx, cy, size)
    healthy, _ = centered(image, cx, cy, size)
    return healthy, mask_crop, crop, component
