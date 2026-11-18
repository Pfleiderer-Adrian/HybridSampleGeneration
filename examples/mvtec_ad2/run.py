"""End-to-end MVTec AD 2 run for one category: split, hybrid generation, downstream.

    python -m examples.mvtec_ad2.run --category can
    python -m examples.mvtec_ad2.run --category can --mixture real_draem
    python -m examples.mvtec_ad2.run --category can --mixture real_anomalydiffusion_hybrid \\
        --anomalydiffusion-manifest <generated>/manifest.json

The split, generator and downstream settings correspond to the V3 benchmark
submissions. Hybrids are generated only for mixtures that draw hybrids.
"""

import argparse
import json

from examples.mvtec_ad2.data.splits import SplitConfiguration, load_or_create_manifest
from examples.mvtec_ad2.downstream.presets import DEFAULT_MIXTURE, MIXTURES, create_downstream_configuration
from examples.mvtec_ad2.downstream.runner import evaluate_downstream, train_downstream
from examples.mvtec_ad2.settings import category_root, study_folder
from examples.mvtec_ad2.synthetic_data_generation.hybrid.pipeline import (
    CATEGORIES,
    category_configuration,
    generate_hybrids,
)

# Grouped 60/40 train/validation split; acquisition variants stay together.
SPLIT = SplitConfiguration(test_fraction=0.0, validation_fraction=0.4, seed=42)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--mixture", default=DEFAULT_MIXTURE, choices=sorted(MIXTURES))
    parser.add_argument("--anomalydiffusion-manifest", help="manifest.json of generated AnomalyDiffusion pairs")
    parser.add_argument("--skip-generation", action="store_true",
                        help="reuse the existing hybrid study instead of generating hybrids")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    config = category_configuration(args.category)
    manifest = load_or_create_manifest(
        study_folder(args.category) / "split_manifest.json", category_root(args.category), SPLIT, grouped=True
    )
    downstream = create_downstream_configuration(
        args.mixture, anomalydiffusion_manifest=args.anomalydiffusion_manifest
    )
    if downstream.data.source_quotas.get("hybrid") and not args.skip_generation:
        generate_hybrids(config, manifest)
    run_folder = train_downstream(config, manifest, downstream)
    metrics = evaluate_downstream(run_folder, manifest)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
