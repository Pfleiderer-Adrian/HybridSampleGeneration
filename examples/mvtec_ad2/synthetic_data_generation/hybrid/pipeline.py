"""Hybrid generation for one MVTec AD 2 category, from ingestion to postprocessing."""

import importlib

from examples.mvtec_ad2.data.dataset import MVTecAD2Dataloader
from examples.mvtec_ad2.data.splits import manifest_samples
from hybrid_sample_generator import HybridDataGenerator

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")


def category_configuration(category: str):
    """Generator configuration of one category (see ``categories/<category>.py``)."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; choose one of {list(CATEGORIES)}.")
    module = importlib.import_module(f"examples.mvtec_ad2.synthetic_data_generation.hybrid.categories.{category}")
    return module.create_configuration()


def generate_hybrids(config, manifest) -> HybridDataGenerator:
    """Run all generator stages on the training partition of ``manifest``."""
    generator = HybridDataGenerator(config)
    generator.ingest_dataset(MVTecAD2Dataloader(manifest_samples(manifest, "train")))
    generator.extract_anomalies()
    generator.train_generator()
    generator.generate_synthetic_anomalies()
    generator.plan_hybrid_samples()
    generator.materialize_hybrid_samples()
    # No-op unless the category enables postprocessing.
    generator.postprocess_hybrid_samples()
    config.save_config_file()
    return generator
