"""Local paths shared by the executable MVTec AD 2 examples."""
import os
from pathlib import Path

DATASET_ROOT = Path(os.environ.get("MVTECAD2_ROOT", "/mnt/results/mvtec2/mvtec_ad_2"))
OUTPUT_ROOT = Path(os.environ.get("MVTECAD2_OUTPUT", "/mnt/results/mvtec2/experiments/testv1.0.3_ohne_hybrid"))
TEXTURE_ROOT = os.environ.get("MVTECAD2_TEXTURES")

def category_root(category: str) -> Path:
    return DATASET_ROOT / category

def study_folder(category: str) -> Path:
    return OUTPUT_ROOT / category / "hybrid_generation"
