"""Run the six remaining comparison_v2 categories in isolated GPU workers."""

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


CATEGORIES = ("fabric", "fruit_jelly", "sheet_metal", "vial", "wallplugs", "walnuts")


def write_status(folder, stage, **values):
    path = folder / "v2_status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        **previous,
        "stage": stage,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }, indent=2) + "\n")
    temporary.replace(path)


def run_worker(root, category):
    import torch
    from hybrid_sample_generator import HybridDataGenerator, load_config_file
    from examples.mvtec_ad2.dataset import MVTecAD2Dataloader
    from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
    from examples.mvtec_ad2.downstream.runner import train_downstream, evaluate_downstream
    from examples.mvtec_ad2.splits import manifest_samples, _validate_manifest

    folder = root / category / "hybrid_generation"
    stage = "initializing"
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Worker requires exactly one visible CUDA GPU.")
        print(f"Category: {category}; GPU: {torch.cuda.get_device_name(0)}; "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        config = load_config_file(folder / "configuration.json")
        downstream = DownstreamConfiguration.load(folder / "downstream_requested.json")
        manifest = json.loads((folder / "split_manifest.json").read_text())
        _validate_manifest(manifest)
        generator = HybridDataGenerator(config)
        steps = [
            ("ingestion", lambda: generator.ingest_dataset(
                MVTecAD2Dataloader(manifest_samples(manifest, "train")))),
            ("extraction", generator.extract_anomalies),
            ("generator_training", generator.train_generator),
            ("synthetic_generation", generator.generate_synthetic_anomalies),
            ("matching", generator.plan_hybrid_samples),
            ("materialization", generator.materialize_hybrid_samples),
        ]
        for stage, operation in steps:
            write_status(folder, stage, pid=os.getpid(), category=category)
            operation()
        config.save_config_file()
        # Release generator weights before allocating DRAEM on the same GPU.
        del steps, operation, generator
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        stage = "downstream_training"
        write_status(folder, stage)
        run_folder = train_downstream(config, manifest, downstream)
        stage = "downstream_evaluation"
        write_status(folder, stage, downstream_run=str(run_folder))
        metrics = evaluate_downstream(run_folder, manifest)
        write_status(folder, "complete", metrics=metrics)
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception as exc:
        write_status(folder, "failed", failed_stage=stage, error=str(exc))
        traceback.print_exc()
        return 1
    return 0


def run_queue(root, split_root):
    import torch
    from examples.mvtec_ad2.common import create_downstream_configuration
    from examples.mvtec_ad2.splits import _validate_manifest
    from hybrid_sample_generator.configuration.study import StudyConfiguration

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Queue requires exactly one visible CUDA GPU.")
    prepared = []
    for category in CATEGORIES:
        folder = root / category / "hybrid_generation"
        if folder.exists():
            raise FileExistsError(f"New study folder required: {folder}")
        module = importlib.import_module(f"examples.mvtec_ad2.categories.{category}")
        config = module.create_configuration()
        config.study = StudyConfiguration(f"comparison_v2_{category}", folder,
                                          seed=config.study.seed)
        config.validate()
        downstream = create_downstream_configuration()
        downstream.validate()
        manifest = json.loads((split_root / category / "split_manifest.json").read_text())
        _validate_manifest(manifest)
        prepared.append((category, folder, config, downstream, manifest))

    for category, folder, config, downstream, manifest in prepared:
        folder.mkdir(parents=True, exist_ok=False)
        config.save_config_file()
        downstream.save(folder / "downstream_requested.json")
        (folder / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write_status(folder, "queued", category=category,
                     gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    # Workers use the snapshots above even if category defaults are edited later.
    del prepared, config, downstream
    failures = []
    for category in CATEGORIES:
        folder = root / category / "hybrid_generation"
        print(f"Starting {category}", flush=True)
        with (folder / "run.log").open("w") as log:
            result = subprocess.run([
                sys.executable, "-u", "-m", "examples.mvtec_ad2.comparison_v2_remaining",
                "--root", str(root), "--category", category,
            ], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(category)
            if json.loads((folder / "v2_status.json").read_text())["stage"] != "failed":
                write_status(folder, "failed", error=f"Worker exited with {result.returncode}")
        print(f"Finished {category}: exit {result.returncode}", flush=True)
    print(f"Queue finished; failed categories: {failures}", flush=True)
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path,
                        default=Path("/mnt/results/mvtec2/experiments/comparison_v1"))
    parser.add_argument("--category", choices=CATEGORIES)
    args = parser.parse_args()
    if args.category:
        return run_worker(args.root.resolve(), args.category)
    return run_queue(args.root.resolve(), args.split_root.resolve())


if __name__ == "__main__":
    sys.exit(main())
