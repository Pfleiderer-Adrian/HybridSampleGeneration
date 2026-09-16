"""Dataset paths, selected categories and split for new MVTec experiments."""

from __future__ import annotations

import os
from pathlib import Path

from examples.mvtec_ad2.configuration import Experiment, SplitConfiguration
from examples.mvtec_ad2.presets import MVTECAD2_CATEGORIES


MVTECAD2_ROOT = Path(
    os.environ.get("MVTECAD2_ROOT", "/mnt/results/mvtec2/mvtec_ad_2")
)
MVTECAD2_SAVE = Path(
    os.environ.get("MVTECAD2_SAVE", "/mnt/results/mvtec2/experiments/test_datarepo_v7")
)

EXPERIMENT = Experiment(
    dataset_root=MVTECAD2_ROOT,
    output_root=MVTECAD2_SAVE,
    categories=MVTECAD2_CATEGORIES,
    split=SplitConfiguration(test_enabled=True, test_fraction=0.2, validation_fraction=0.2),
)
