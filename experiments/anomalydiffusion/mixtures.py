"""Declared V3 source budgets, with a fixed 400-draw synthetic budget."""

from copy import deepcopy


PROFILES = {
    "real_anomalydiffusion": ({"normal": 500, "real": 100, "anomalydiffusion": 400}, None),
    "real_anomalydiffusion_hybrid": (
        {"normal": 500, "real": 100, "anomalydiffusion": 200, "hybrid": 200}, None),
    "real_anomalydiffusion_draem": (
        {"normal": 500, "real": 100, "anomalydiffusion": 200, "draem": 200}, None),
    "real_anomalydiffusion_draem_hybrid": (
        {"normal": 500, "real": 100, "anomalydiffusion": 134, "draem": 133, "hybrid": 133},
        ["anomalydiffusion", "draem", "hybrid"]),
}


def profile(variant):
    """Return independent configuration values for one declared experiment."""
    return deepcopy(PROFILES[variant])
