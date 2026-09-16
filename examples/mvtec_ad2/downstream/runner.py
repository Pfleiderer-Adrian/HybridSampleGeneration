"""Train and evaluate DRAEM for an already prepared MVTec use case."""

from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4
import torch

from hybrid_sample_generator.persistence.study_repository import StudyRepository
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from examples.mvtec_ad2.splits import manifest_samples
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from .datasets import HybridPairs, MixedTrainingDataset, RealImageDataset
from .draem.model import DRAEM
from .evaluation import evaluate, evaluation_loader
from .training import train
from .textures import prepare_textures


def run_folder(study, run_id):
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Evaluation requires a concrete downstream_run_id (directory name).")
    return Path(study.config.study.folder) / "downstream" / "draem" / run_id


def train_downstream(study, *, hybrids=None, save_study_config=True):
    config, manifest = deepcopy(study.config.downstream), study.split_manifest
    if config is None or manifest is None:
        raise ValueError("Downstream training requires configuration and a saved split.")
    config.validate()
    folder = Path(study.config.study.folder)
    if config.data.hybrid_fraction > 0 and hybrids is None:
        database = study.config.study.paths.artifact_database
        if not Path(database).is_file():
            raise ValueError("No hybrid study. Select generation/materialization steps first.")
        hybrids = HybridPairs(StudyRepository(database), ArtifactStore(folder), manifest, config.data)
    prepare_textures(folder, config.data)
    if save_study_config:
        study.config.save_config_file()
    dataset = MixedTrainingDataset(manifest_samples(manifest, "train", healthy_only=True), hybrids, config)
    validation = RealImageDataset(manifest_samples(manifest, "validation"), config.data)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = run_folder(study, run_id)
    output.mkdir(parents=True)
    config.save(output / "configuration.json")
    used_manifest = {**manifest, "hybrid_provenance": hybrids.provenance if hybrids is not None else []}
    (output / "split_manifest.json").write_text(json.dumps(used_manifest, indent=2) + "\n")
    train(dataset, validation, config, output)
    print(f"DRAEM run: {output}")
    return output


def load_run(study, run_id):
    """Validate run files and split identity without loading model weights."""
    output = run_folder(study, run_id)
    for name in ("configuration.json", "split_manifest.json", "checkpoints/best.pt"):
        if not (output / name).is_file():
            raise ValueError(f"Downstream run is missing {name}: {output}")
    config = DownstreamConfiguration.load(output / "configuration.json")
    manifest = json.loads((output / "split_manifest.json").read_text())
    if study.split_manifest is None or manifest["fingerprint"] != study.split_manifest["fingerprint"]:
        raise ValueError("Downstream run belongs to a different split.")
    if {key: value for key, value in manifest.items() if key != "hybrid_provenance"} != study.split_manifest:
        raise ValueError("Downstream run split manifest has been modified.")
    return output, config, manifest


def evaluate_downstream(study, run_id):
    output, config, manifest = load_run(study, run_id)
    settings = config.training
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if settings.device == "auto" else settings.device)
    model = DRAEM(settings.reconstruction_width, settings.segmentation_width).to(device)
    model.load_state_dict(torch.load(output / "checkpoints" / "best.pt", map_location=device, weights_only=True)["model"])
    metrics = {"protocol": "custom_stratified_anomaly_supervised", "test_enabled": manifest["split"]["test_enabled"]}
    partitions = ["validation", "test"] if metrics["test_enabled"] else ["validation"]
    for partition in partitions:
        dataset = RealImageDataset(manifest_samples(manifest, partition), config.data)
        loader = evaluation_loader(dataset, settings)
        metrics[partition] = evaluate(model, loader, device, output / "predictions" / partition,
                                      patch_batch_size=settings.batch_size)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"DRAEM evaluation: {output / 'metrics.json'}")
    return output
