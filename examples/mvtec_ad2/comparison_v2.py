"""Create a fresh comparison_v2 can study and retrain/evaluate DRAEM.

Run from the repository root: python -m examples.mvtec_ad2.comparison_v2
"""

import argparse
import json
from pathlib import Path

from examples.mvtec_ad2.categories.can import CATEGORY, create_configuration
from examples.mvtec_ad2.common import create_downstream_configuration
from examples.mvtec_ad2.dataset import MVTecAD2Dataloader
from examples.mvtec_ad2.downstream.runner import evaluate_downstream, train_downstream
from examples.mvtec_ad2.settings import OUTPUT_ROOT, category_root
from examples.mvtec_ad2.splits import (
    SplitConfiguration,
    load_or_create_manifest,
    manifest_samples,
)
from hybrid_sample_generator import HybridDataGenerator
from hybrid_sample_generator.configuration.study import StudyConfiguration


STUDY_NAME = "comparison_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-folder",
        type=Path,
        default=OUTPUT_ROOT / STUDY_NAME / CATEGORY / "hybrid_generation",
        help="New study directory; must not already exist.",
    )
    args = parser.parse_args()
    folder = args.study_folder.expanduser().resolve()
    if folder.exists():
        parser.error(f"Study directory already exists: {folder}. Choose a new --study-folder.")

    # Rebuild settings from the current can.py/common.py, never a saved config.
    config = create_configuration()
    config.study = StudyConfiguration(STUDY_NAME, folder, seed=config.study.seed)
    downstream_config = create_downstream_configuration()
    config.validate()
    downstream_config.validate()

    manifest = load_or_create_manifest(
        folder / "split_manifest.json",
        category_root(CATEGORY),
        SplitConfiguration(test_fraction=0.2, validation_fraction=0.2, seed=42),
    )
    config.save_config_file()
    print(f"Study: {config.study.name}\nOutput: {folder}", flush=True)

    generator = HybridDataGenerator(config)
    generator.ingest_dataset(MVTecAD2Dataloader(manifest_samples(manifest, "train")))
    generator.extract_anomalies()
    generator.train_generator()
    generator.generate_synthetic_anomalies()
    generator.plan_hybrid_samples()
    generator.materialize_hybrid_samples()
    config.save_config_file()

    run_folder = train_downstream(config, manifest, downstream_config)
    metrics = evaluate_downstream(run_folder, manifest)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
