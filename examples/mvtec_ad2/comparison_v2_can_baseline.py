"""Train CAN's V2 baseline from the saved hybrid-run settings and split."""

import json
import os
from pathlib import Path
import traceback

from hybrid_sample_generator import load_config_file
from hybrid_sample_generator.configuration.study import StudyConfiguration
from examples.mvtec_ad2.comparison_v2_remaining import write_status
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.runner import train_downstream, evaluate_downstream
from examples.mvtec_ad2.splits import _validate_manifest


def main():
    import torch

    category = Path("/mnt/results/mvtec2/experiments/comparison_v2/can")
    folder = category / "baseline"
    hybrid_run = category / "hybrid_generation/downstream/draem/20261002T083122Z_7f66e13f"
    if folder.exists():
        raise FileExistsError(f"Baseline folder already exists: {folder}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required.")

    config = load_config_file(category / "hybrid_generation/configuration.json")
    config.study = StudyConfiguration("comparison_v2_can_baseline", folder,
                                      seed=config.study.seed)
    downstream = DownstreamConfiguration.load(hybrid_run / "configuration.json")
    downstream.data.hybrid_fraction = 0.0
    downstream.validate()
    manifest = json.loads((category / "hybrid_generation/split_manifest.json").read_text())
    _validate_manifest(manifest)
    hybrid_manifest = json.loads((hybrid_run / "split_manifest.json").read_text())
    if {k: v for k, v in hybrid_manifest.items() if k != "hybrid_provenance"} != manifest:
        raise ValueError("Baseline split differs from the hybrid run.")
    folder.mkdir(parents=True, exist_ok=False)
    config.save_config_file()
    downstream.save(folder / "downstream_requested.json")
    (folder / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stage = "downstream_training"
    try:
        write_status(folder, stage, pid=os.getpid(),
                     gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                     hybrid_reference=str(hybrid_run))
        print(f"CAN V2 baseline: {folder}; GPU: {torch.cuda.get_device_name(0)}", flush=True)
        run = train_downstream(config, manifest, downstream)
        stage = "downstream_evaluation"
        write_status(folder, stage, downstream_run=str(run))
        metrics = evaluate_downstream(run, manifest)
        write_status(folder, "complete", metrics=metrics)
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception as exc:
        write_status(folder, "failed", failed_stage=stage, error=str(exc))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
