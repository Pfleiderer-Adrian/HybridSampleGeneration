"""Read-only resolution/visibility audit of the exact frozen V3 hybrid pools."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.ndimage import find_objects, label, zoom
from PIL import Image

from hybrid_sample_generator.imaging.resampling import resize_and_pad
from .prepare import HYBRID_REFERENCE, write_json


def spatial(array):
    return np.any(array > 0, axis=0) if array.ndim == 3 else array > 0


def load(study, path):
    return np.load(study / path, mmap_mode="r", allow_pickle=False)


def mask_roundtrip(mask, target):
    """Diagnostic on the extraction grids, not a claim about synthesis fidelity."""
    scale = min(1., *(t / s for t, s in zip(target, mask.shape)))
    small = zoom(mask.astype(np.uint8), scale, order=0) if scale < 1 else mask.copy()
    restored = zoom(small, tuple(s / n for s, n in zip(mask.shape, small.shape)), order=0) if small.size else np.zeros_like(mask)
    lost = int(np.count_nonzero(mask & ~restored.astype(bool)))
    return scale, int(np.count_nonzero(small)), lost


def expected_placement(mask, metadata, position, shape):
    """Match fusion cropping, inverse scale, boundary shifting, and clipping."""
    if not mask.any():
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0), bool)
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    factor = 1 / np.asarray(metadata["scale_factor"], dtype=float)
    restored = zoom(crop.astype(np.uint8), factor, order=0).astype(bool)
    offset = np.array([round(n * p - s / 2) for n, p, s in zip(shape, position, restored.shape)])
    end = offset + restored.shape
    for axis in range(2):
        if end[axis] > shape[axis]:
            offset[axis] -= end[axis] - shape[axis]
        if offset[axis] < 0:
            offset[axis] = 0
    size = tuple(min(s, n - int(o)) for s, n, o in zip(restored.shape, shape, offset))
    slices = tuple(slice(int(o), int(o + s)) for o, s in zip(offset, size))
    return slices, restored[:size[0], :size[1]]


def audit_conditioning(study, output):
    """Check raw annotation identity and mask support at actual SPADE scales."""
    import torch
    con = sqlite3.connect(f"file:{study}/artifacts.sqlite?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    originals = [dict(r) for r in con.execute("SELECT * FROM original_samples WHERE has_anomaly=1")]
    records = [dict(r) for r in con.execute("SELECT * FROM synthetic_anomalies")]
    con.close()
    database = next(study.glob("mvtecad2_*.db"))
    con = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    direction = con.execute("SELECT direction FROM study_directions").fetchone()[0]
    order = "ASC" if direction == "MINIMIZE" else "DESC"
    number, trial_id = con.execute(f"SELECT t.number,t.trial_id FROM trials t JOIN trial_values v USING(trial_id) WHERE t.state='COMPLETE' ORDER BY v.value {order} LIMIT 1").fetchone()
    attrs = dict(con.execute("SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?", (trial_id,)))
    params = json.loads(attrs["params"])
    con.close()
    levels = params["n_levels"]
    changed, removed_pixels = [], 0
    for row in originals:
        meta = json.loads(row["metadata_json"])
        raw = np.array(Image.open(meta["source_segmentation_path"]).convert("L")) > 0
        stored = spatial(load(study, row["segmentation_path"]))
        if not np.array_equal(raw, stored):
            changed.append(row["source_name"])
            removed_pixels += int((raw & ~stored).sum())
    empty = Counter()
    for row in records:
        mask = spatial(load(study, row["segmentation_path"]))
        tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        size = mask.shape[0]
        # Encoder reduces by 2**levels. Decoder conditions after each upsample,
        # including the full generator resolution in the last SPADE stage.
        for stage in range(levels):
            width = size // (2 ** (levels - stage - 1))
            reduced = torch.nn.functional.interpolate(tensor, (width, width), mode="nearest")
            empty[str(width)] += not bool(reduced.any())
    result = {"best_trial": number, "n_levels": levels, "synthetic_masks": len(records),
              "empty_conditioning_masks_by_spade_resolution": dict(empty),
              "raw_annotations_compared": len(originals), "raw_vs_stored_mask_mismatches": changed,
              "raw_pixels_removed_before_extraction": removed_pixels,
              "interpretation": "Empty coarse conditioning does not imply complete loss: the last SPADE stage receives the full-resolution mask."}
    write_json(output / "conditioning.json", result)
    return result


def audit_category(category, study, frozen_run, output):
    config = json.loads((study / "configuration.json").read_text())
    frozen_split = json.loads((frozen_run / "split_manifest.json").read_text())
    provenance = frozen_split["hybrid_provenance"]
    used_hybrids = {r["hybrid_id"] for r in provenance}
    train_ids = {r["sample_id"] for r in frozen_split["partitions"]["train"]}
    con = sqlite3.connect(f"file:{study}/artifacts.sqlite?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    tables = {name: [dict(r) for r in con.execute(f"SELECT * FROM {name}")]
              for name in ("original_samples", "real_anomalies", "synthetic_anomalies", "hybrid_samples", "placements")}
    con.close()
    originals = {r["id"]: r for r in tables["original_samples"]}
    real = {r["id"]: r for r in tables["real_anomalies"]}
    by_original = defaultdict(list)
    for row in real.values():
        by_original[row["original_sample_id"]].append(row)
    synthetic = {r["id"]: r for r in tables["synthetic_anomalies"]}
    placements = defaultdict(list)
    for row in tables["placements"]:
        placements[row["hybrid_sample_id"]].append(row)
    target = tuple(config["extraction"]["anomaly_size"][-2:])
    threshold = int(config["extraction"]["min_coverage_ratio"] * np.prod(target))
    summary = Counter()
    components, synthetic_rows, hybrid_rows, placement_rows, visibility = [], [], [], [], []
    for original in originals.values():
        if original["source_name"] not in train_ids:
            raise ValueError(f"Non-training original in study: {original['source_name']}")
        if not original["has_anomaly"]:
            continue
        original_mask = spatial(load(study, original["segmentation_path"]))
        labeled, n = label(original_mask)
        regions = find_objects(labeled)
        extracted = sorted(by_original[original["id"]], key=lambda r: r["component_index"])
        kept_index = 0
        summary["anomalous_originals"] += 1
        for number, region in enumerate(regions, start=1):
            mask = labeled[region] == number
            pixels = int(mask.sum())
            omitted = pixels < threshold
            row = {"original_id": original["id"], "sample_id": original["source_name"],
                   "source_component": number, "pixels": pixels,
                   "bbox_yxhw": [region[0].start, region[1].start, *mask.shape],
                   "omitted_by_size_filter": omitted}
            summary["original_components"] += 1
            summary["original_mask_pixels"] += pixels
            if omitted:
                summary["omitted_components"] += 1
                summary["omitted_mask_pixels"] += pixels
            else:
                record = extracted[kept_index]
                kept_index += 1
                normalized = spatial(load(study, record["segmentation_path"]))
                expected, _ = resize_and_pad(mask[None].astype(np.uint8), target,
                                             order=0, preserve_aspect_ratio=True, padding_value=0)
                metadata = json.loads(record["metadata_json"])
                center = [round((r.start + r.stop - 1) / 2) for r in region]
                if center != metadata["centroid_voxel"] or not np.array_equal(expected[0] > 0, normalized):
                    raise ValueError(f"Stored extraction differs from reconstructed source: {record['id']}")
                scale, normalized_pixels, lost = mask_roundtrip(mask, target)
                row.update(real_id=record["id"], scale=scale, normalized_pixels=normalized_pixels,
                           roundtrip_lost_pixels=lost, roundtrip_recall=1 - lost / pixels)
                summary["extracted_components"] += 1
                summary["normalized_empty_components"] += not normalized.any()
                summary["downscaled_components"] += scale < 1
                summary["roundtrip_lost_pixels"] += lost
                summary["retained_source_pixels"] += pixels
            components.append(row)
        if kept_index != len(extracted):
            raise ValueError("Stored extraction count differs from the reconstructed size filter.")
        summary["originals_without_extracted_component"] += kept_index == 0
    for item in synthetic.values():
        mask = spatial(load(study, item["segmentation_path"]))
        pixels = int(mask.sum())
        synthetic_rows.append({"synthetic_id": item["id"], "real_id": item["real_anomaly_id"], "pixels": pixels})
        summary["synthetic_masks"] += 1
        summary["synthetic_empty_masks"] += pixels == 0
    # Background ordering reuses memory maps for sibling hybrids.
    hybrids = sorted((r for r in tables["hybrid_samples"] if r["id"] in used_hybrids),
                     key=lambda r: (r["original_sample_id"], r["id"]))
    if {r["id"] for r in hybrids} != used_hybrids:
        raise ValueError("Missing hybrids from the frozen downstream provenance.")
    current = None
    for hybrid in hybrids:
        original = originals[hybrid["original_sample_id"]]
        if current != original["id"]:
            healthy = load(study, original["image_path"])
            current = original["id"]
        mask = spatial(load(study, hybrid["segmentation_path"]))
        image = load(study, hybrid["image_path"])
        if image.shape != healthy.shape or mask.shape != image.shape[-2:]:
            raise ValueError("Unaligned materialized hybrid.")
        pixels = int(mask.sum())
        summary["used_hybrids"] += 1
        summary["hybrid_empty_masks"] += pixels == 0
        labeled, n = label(mask)
        hybrid_rows.append({"hybrid_id": hybrid["id"], "mask_pixels": pixels, "components": n})
        for component_index, region in enumerate(find_objects(labeled), start=1):
            support = labeled[region] == component_index
            count = int(support.sum())
            # Compare persisted hybrid against its exact healthy recipient only on
            # labeled pixels; avoids mistaking a nonempty label for a visible defect.
            image_crop = image[(slice(None), *region)][:, support].astype(np.float32)
            healthy_crop = healthy[(slice(None), *region)][:, support].astype(np.float32)
            delta = np.max(np.abs(image_crop - healthy_crop), axis=0)
            max_delta, mean_delta = float(delta.max()), float(delta.mean())
            visibility.append({"hybrid_id": hybrid["id"], "component": component_index,
                               "pixels": count, "max_channel_delta": max_delta,
                               "mean_max_channel_delta": mean_delta,
                               "bbox_yxhw": [region[0].start, region[1].start, *support.shape],
                               "fraction_delta_lt_1": float((delta < 1).mean())})
            summary["final_components"] += 1
            summary["exactly_unchanged_components"] += max_delta <= 1e-5
            summary["components_max_delta_lt_1"] += max_delta < 1
            summary["final_small_components_le_16"] += count <= 16
            summary["unchanged_small_components_le_16"] += count <= 16 and max_delta <= 1e-5
        for placement in placements[hybrid["id"]]:
            syn = synthetic[placement["synthetic_anomaly_id"]]
            record = real[syn["real_anomaly_id"]]
            slices, expected = expected_placement(spatial(load(study, syn["segmentation_path"])),
                json.loads(record["metadata_json"]), (placement["position_y"], placement["position_x"]), mask.shape)
            expected_pixels = int(expected.sum())
            full = study / "artifacts/placements" / placement["id"] / "full_segmentation.npy"
            own = spatial(np.load(full, mmap_mode="r"))[slices] if full.exists() else mask[slices]
            overlap = int((expected & own).sum())
            exact = full.exists()
            placement_rows.append({"placement_id": placement["id"], "hybrid_id": hybrid["id"],
                "real_id": record["id"], "expected_pixels_before_alpha": expected_pixels,
                "retained_expected_pixels": overlap, "own_placement_mask_available": exact,
                "comparison": "own_stored_mask" if exact else "final_union_upper_bound"})
            summary["used_placements"] += 1
            summary["placements_with_own_full_mask"] += exact
            summary["placements_zero_overlap_with_final_mask"] += overlap == 0
    summary["all_generated_hybrids"] = sum(r["status"] == "generated" for r in tables["hybrid_samples"])
    summary["minimum_original_component_pixels"] = min(r["pixels"] for r in components)
    result = {"category": category, "study": str(study), "frozen_run": str(frozen_run),
              "target_size": target, "minimum_component_pixels": threshold, "counts": dict(summary),
              "omitted_pixel_fraction": summary["omitted_mask_pixels"] / summary["original_mask_pixels"],
              "roundtrip_lost_pixel_fraction": summary["roundtrip_lost_pixels"] / summary["retained_source_pixels"],
              "minimum_extraction_scale": min(r.get("scale", 1) for r in components),
              "source_size_bins": {name: sum(lo <= r["pixels"] <= hi for r in components)
                  for name, lo, hi in (("1_3", 1, 3), ("4_15", 4, 15), ("16_63", 16, 63),
                                       ("64_255", 64, 255), ("256_plus", 256, 10**12))}}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary", result), ("original_components", components),
                        ("synthetic_masks", synthetic_rows), ("hybrids", hybrid_rows),
                        ("placements", placement_rows), ("visibility", visibility)):
        write_json(output / f"{name}.json", value)
    audit_conditioning(study, output)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+")
    args = parser.parse_args()
    reference = json.loads((HYBRID_REFERENCE / "manifest.json").read_text())
    summaries = []
    for category, job in reference["jobs"].items():
        if args.categories and category not in args.categories:
            continue
        status = json.loads(Path(job["selection_status"]).read_text())
        summaries.append(audit_category(category, Path(status["hybrid_study"]), Path(job["run"]), args.output / category))
    write_json(args.output / "summary.json", summaries)


if __name__ == "__main__":
    main()
