"""Run the seven missing V2 baselines sequentially on one visible GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from examples.mvtec_ad2.comparison_v2_remaining import write_status


CATEGORIES = ("rice", "fabric", "fruit_jelly", "sheet_metal", "vial", "wallplugs", "walnuts")


def prepare(root, dry_run=False):
    from hybrid_sample_generator import load_config_file
    from hybrid_sample_generator.configuration.study import StudyConfiguration
    from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
    from examples.mvtec_ad2.downstream.textures import TextureSynthesizer
    from examples.mvtec_ad2.splits import _validate_manifest

    shared_textures = root / "rice/hybrid_generation/downstream/textures/dtd/images"
    prepared = []
    for category in CATEGORIES:
        folder = root / category / "baseline"
        source = root / category / "hybrid_generation"
        if folder.exists():
            raise FileExistsError(f"Baseline folder already exists: {folder}")
        config = load_config_file(source / "configuration.json")
        config.study = StudyConfiguration(f"comparison_v2_{category}_baseline", folder,
                                          seed=config.study.seed)
        runs = sorted((source / "downstream/draem").glob("*/configuration.json"))
        if len(runs) > 1:
            raise ValueError(f"Ambiguous hybrid reference for {category}: {runs}")
        reference = runs[0] if runs else source / "downstream_requested.json"
        downstream = DownstreamConfiguration.load(reference)
        downstream.data.hybrid_fraction = 0.0
        # Reuse the same DTD corpus for hybrid runs whose textures are not prepared yet.
        if downstream.data.texture_root is None:
            downstream.data.texture_root = str(shared_textures)
        TextureSynthesizer(downstream.data.texture_root)
        downstream.validate()
        manifest = json.loads((source / "split_manifest.json").read_text())
        _validate_manifest(manifest)
        if runs:
            saved = json.loads((reference.parent / "split_manifest.json").read_text())
            if {k: v for k, v in saved.items() if k != "hybrid_provenance"} != manifest:
                raise ValueError(f"Hybrid split mismatch for {category}")
        prepared.append((category, folder, config, downstream, manifest, reference))
        print(f"Validated {category}: {downstream.training.epochs} epochs; "
              f"reference {reference}", flush=True)
    if dry_run:
        return
    for category, folder, config, downstream, manifest, reference in prepared:
        folder.mkdir(parents=True, exist_ok=False)
        config.save_config_file()
        downstream.save(folder / "downstream_requested.json")
        (folder / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write_status(folder, "queued", category=category,
                     gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                     hybrid_reference=str(reference))


def worker(root, category):
    import torch
    from hybrid_sample_generator import load_config_file
    from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
    from examples.mvtec_ad2.downstream.runner import train_downstream, evaluate_downstream

    folder = root / category / "baseline"
    stage = "downstream_training"
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible CUDA GPU is required.")
        config = load_config_file(folder / "configuration.json")
        downstream = DownstreamConfiguration.load(folder / "downstream_requested.json")
        manifest = json.loads((folder / "split_manifest.json").read_text())
        write_status(folder, stage, pid=os.getpid())
        print(f"Baseline {category}: GPU {torch.cuda.get_device_name(0)}; "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        run = train_downstream(config, manifest, downstream)
        stage = "downstream_evaluation"
        write_status(folder, stage, downstream_run=str(run))
        metrics = evaluate_downstream(run, manifest)
        write_status(folder, "complete", metrics=metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        return 0
    except Exception as exc:
        write_status(folder, "failed", failed_stage=stage, error=str(exc))
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.category:
        return worker(root, args.category)
    prepare(root, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    failures = []
    for category in CATEGORIES:
        folder = root / category / "baseline"
        print(f"Starting baseline {category}", flush=True)
        with (folder / "run.log").open("w") as log:
            result = subprocess.run([
                sys.executable, "-u", "-m", "examples.mvtec_ad2.comparison_v2_baselines",
                "--root", str(root), "--category", category,
            ], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(category)
            if json.loads((folder / "v2_status.json").read_text())["stage"] != "failed":
                write_status(folder, "failed", error=f"Worker exit {result.returncode}")
        print(f"Finished baseline {category}: exit {result.returncode}", flush=True)
    print(f"Baseline queue finished; failures: {failures}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
