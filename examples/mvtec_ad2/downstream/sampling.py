"""Exact per-epoch source quotas with reproducible ordering."""

import numpy as np


def source_plan(data, seed, epoch):
    normal = round(data.samples_per_epoch * data.normal_fraction)
    anomalous = data.samples_per_epoch - normal
    hybrid = round(anomalous * data.hybrid_fraction)
    perlin = anomalous - hybrid
    if anomalous == 0:
        raise ValueError("samples_per_epoch is too small for the requested mixture.")
    for fraction, count in ((data.hybrid_fraction, hybrid), (1 - data.hybrid_fraction, perlin)):
        if fraction > 0 and count == 0:
            raise ValueError("Increase samples_per_epoch to represent every requested source.")
    plan = ["normal"] * normal + ["hybrid"] * hybrid + ["draem"] * perlin
    np.random.default_rng(np.random.SeedSequence([seed, epoch])).shuffle(plan)
    return plan
