"""Masks and healthy backgrounds from a materialized hybrid study (SQLite + NumPy only).

AnomalyDiffusion reuses the defect masks of the hybrid study together with the
healthy training image of each hybrid. This module reads the study directly so
that it also runs in the isolated AnomalyDiffusion environment.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np

from .geometry import crop_around_component


def _hybrid_ids(path):
    """IDs from a JSON list, a provenance list or a downstream ``split_manifest.json``."""
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        value = value["hybrid_provenance"]
    ids = [item["hybrid_id"] if isinstance(item, dict) else item for item in value]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Hybrid ID selection must be nonempty and unique.")
    return ids


def hybrid_sources(study, split, hybrid_ids=None):
    """Eligible hybrids of ``study``: generated, on healthy training originals.

    Without ``hybrid_ids`` all eligible hybrids are used in ID order; hybrids
    rejected by postprocessing with action ``exclude`` are skipped, matching
    the downstream loader. ``hybrid_ids`` (a path, see ``_hybrid_ids``) selects
    and orders hybrids explicitly, e.g. the provenance of a previous run.
    """
    study = Path(study)
    train = {r["sample_id"]: r for r in split["partitions"]["train"]}
    connection = sqlite3.connect(f"file:{study / 'artifacts.sqlite'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        originals = {r["id"]: dict(r) for r in connection.execute("SELECT * FROM original_samples")}
        hybrids = {r["id"]: dict(r) for r in connection.execute("SELECT * FROM hybrid_samples")}
        donors = {}
        for r in connection.execute(
                "SELECT p.hybrid_sample_id AS hybrid, r.original_sample_id AS donor FROM placements p "
                "JOIN synthetic_anomalies s ON s.id = p.synthetic_anomaly_id "
                "JOIN real_anomalies r ON r.id = s.real_anomaly_id"):
            donors.setdefault(r["hybrid"], set()).add(r["donor"])
        rejected = set()
        configuration = study / "configuration.json"
        post = json.loads(configuration.read_text()).get("postprocessing", {}) if configuration.is_file() else {}
        has_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='postprocessing_decisions'").fetchone()
        if post.get("enabled") and post.get("action") == "exclude" and has_table:
            rejected = {r[0] for r in connection.execute(
                "SELECT hybrid_sample_id FROM postprocessing_decisions WHERE accepted = 0")}
    finally:
        connection.close()

    def eligible(hybrid_id):
        h = hybrids[hybrid_id]
        o = originals[h["original_sample_id"]]
        return (h["status"] == "generated" and not o["has_anomaly"] and o["source_name"] in train
                and train[o["source_name"]]["label"] == "good"
                and all(originals[d]["source_name"] in train for d in donors.get(hybrid_id, ())))

    if hybrid_ids is None:
        selected = [i for i in sorted(hybrids) if i not in rejected and eligible(i)]
    else:
        selected = _hybrid_ids(hybrid_ids)
        unknown = [i for i in selected if i not in hybrids or not eligible(i)]
        if unknown:
            raise ValueError(f"Selected hybrids are missing or ineligible: {unknown[:5]}")
    if not selected:
        raise ValueError("No eligible hybrids on healthy training originals.")
    return [dict(hybrid_id=i, original_id=hybrids[i]["original_sample_id"],
                 donor_original_ids=sorted(donors.get(i, ())),
                 segmentation_path=hybrids[i]["segmentation_path"],
                 image_path=originals[hybrids[i]["original_sample_id"]]["image_path"],
                 recipient_sample_id=originals[hybrids[i]["original_sample_id"]]["source_name"])
            for i in selected]


def control_and_mask(study, source, rng):
    """Healthy 256 crop and binary mask around one random component of a hybrid."""
    study = Path(study)
    mask = np.any(np.load(study / source["segmentation_path"], allow_pickle=False) > 0, axis=0)
    image = np.load(study / source["image_path"], allow_pickle=False)
    image = np.clip(image.transpose(1, 2, 0), 0, 255).round().astype("uint8")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] != 3:
        raise ValueError("Expected a grayscale or RGB healthy original.")
    healthy, native_mask, crop, component = crop_around_component(image, mask, rng)
    return healthy, native_mask, dict(recipient_sample_id=source["recipient_sample_id"], crop_xywh=crop,
                                      hybrid_id=source["hybrid_id"],
                                      donor_original_ids=source["donor_original_ids"],
                                      selected_component=component)
