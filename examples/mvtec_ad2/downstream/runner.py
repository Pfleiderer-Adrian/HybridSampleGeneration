"""Train and evaluate DRAEM from explicit generator and split inputs."""
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from examples.mvtec_ad2.splits import manifest_samples
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.study_repository import StudyRepository

from .configuration import DownstreamConfiguration
from .datasets import HybridPairs, MixedTrainingDataset, RealImageDataset
from .draem.model import DRAEM
from .evaluation import evaluate, evaluation_loader
from .textures import prepare_textures
from .training import train


def train_downstream(generator_config, manifest, config: DownstreamConfiguration):
    """Train DRAEM using materialized hybrids and the persisted training split."""
    config = deepcopy(config)
    config.validate()
    study_folder = Path(generator_config.study.folder)
    hybrids = None
    if config.data.hybrid_fraction > 0:
        database = Path(generator_config.study.paths.artifact_database)
        if not database.is_file():
            raise ValueError("No materialized hybrid study found.")
        repository = StudyRepository(database)
        store = ArtifactStore(study_folder)
        hybrids = HybridPairs(repository, store, manifest, config.data)
    prepare_textures(study_folder, config.data)
    dataset = MixedTrainingDataset(
        manifest_samples(manifest, "train", healthy_only=True), hybrids, config
    )
    validation = RealImageDataset(manifest_samples(manifest, "validation"), config.data)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = study_folder / "downstream" / "draem" / run_id
    output.mkdir(parents=True)
    config.save(output / "configuration.json")
    used_manifest = {
        **manifest,
        "hybrid_provenance": hybrids.provenance if hybrids is not None else [],
    }
    (output / "split_manifest.json").write_text(json.dumps(used_manifest, indent=2) + "\n")
    train(dataset, validation, config, output)
    print(f"DRAEM run: {output}")
    return output


def load_run(run_folder, manifest):
    """Validate one concrete run folder against the active split."""
    output = Path(run_folder)
    for name in ("configuration.json", "split_manifest.json", "checkpoints/best.pt"):
        if not (output / name).is_file():
            raise ValueError(f"Downstream run is missing {name}: {output}")
    config = DownstreamConfiguration.load(output / "configuration.json")
    saved_manifest = json.loads((output / "split_manifest.json").read_text())
    if saved_manifest.get("fingerprint") != manifest.get("fingerprint"):
        raise ValueError("Downstream run belongs to a different split.")
    comparable = {key: value for key, value in saved_manifest.items() if key != "hybrid_provenance"}
    if comparable != manifest:
        raise ValueError("Downstream run split manifest has been modified.")
    return output, config, saved_manifest


def evaluate_downstream(run_folder, manifest):
    """Evaluate a trained DRAEM run on validation and held-out test samples."""
    output, config, saved_manifest = load_run(run_folder, manifest)
    settings = config.training
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name if settings.device == "auto" else settings.device)
    model = DRAEM(settings.reconstruction_width, settings.segmentation_width).to(device)
    checkpoint = torch.load(output / "checkpoints" / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    metrics = {"protocol": "custom_stratified_anomaly_supervised"}
    partitions = ("validation", "test") if saved_manifest["partitions"]["test"] else ("validation",)
    for partition in partitions:
        dataset = RealImageDataset(manifest_samples(saved_manifest, partition), config.data)
        loader = evaluation_loader(dataset, settings)
        metrics[partition] = evaluate(
            model, loader, device, output / "predictions" / partition,
            patch_batch_size=settings.batch_size,
        )
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"DRAEM evaluation: {metrics_path}")
    return metrics
