"""Similarity measures shared by matching and generation workflows."""

import numpy as np


def ssim_01(x, y, data_range=None, k1=0.01, k2=0.03):
    """Return a compact global SSIM score clipped to the interval ``[0, 1]``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim >= 3:
        return float(
            np.mean([ssim_01(a, b, data_range, k1, k2) for a, b in zip(x, y)])
        )
    if data_range is None:
        data_range = max(x.max(), y.max()) - min(x.min(), y.min())
        if data_range == 0:
            return 1.0 if np.allclose(x, y) else 0.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = ((x - mu_x) * (y - mu_y)).mean()
    value = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(np.clip(value, 0.0, 1.0))


__all__ = ["ssim_01"]
