"""Runtime paths for the MVTec AD 2 example."""

from __future__ import annotations

import os
from pathlib import Path


MVTECAD2_ROOT = Path(
    os.environ.get("MVTECAD2_ROOT", "/mnt/results/mvtec2/mvtec_ad_2")
)
MVTECAD2_SAVE = Path(
    os.environ.get("MVTECAD2_SAVE", "/mnt/results/mvtec2/experiments/test_datarepo_v4")
)
