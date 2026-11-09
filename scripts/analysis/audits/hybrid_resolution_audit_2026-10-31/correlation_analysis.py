"""Descriptive category-level correlations; no causal inference or independent runs."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import audit_dir

from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = audit_dir('hybrid_resolution_audit_2026-10-31')
# User-reported external V3 scores, documented in reports/external_submissions_2026-10-31.md.
SCORES = {
    "can": (56.86, 48.40), "fabric": (82.30, 82.92),
    "fruit_jelly": (81.74, 83.27), "rice": (75.60, 80.98),
    "sheet_metal": (73.35, 78.17), "vial": (50.32, 57.88),
    "wallplugs": (55.90, 54.36), "walnuts": (79.99, 81.59),
}


def main():
    audit = json.loads((ROOT / "summary.json").read_text())
    assert {r["category"] for r in audit} == set(SCORES)
    rows = []
    for item in audit:
        category, counts = item["category"], item["counts"]
        hybrid, draem = SCORES[category]
        omitted = counts.get("omitted_components", 0)
        ghosts = counts["exactly_unchanged_components"]
        hybrids = json.loads((ROOT / category / "hybrids.json").read_text())
        visibility = json.loads((ROOT / category / "visibility.json").read_text())
        ghost_pixels = Counter()
        for region in visibility:
            if region["max_channel_delta"] <= 1e-5:
                ghost_pixels[region["hybrid_id"]] += region["pixels"]
        # Matches the sampling distribution: uniform hybrid, then uniform positive
        # mask pixel as the crop anchor. This is an expectation, not a replay of
        # seeded training. A ghost anchor can still yield a crop with a real defect.
        anchor_probability = sum(ghost_pixels[r["hybrid_id"]] / r["mask_pixels"] for r in hybrids) / len(hybrids)
        rows.append({
            "category": category, "external_real_hybrid": hybrid,
            "external_real_draem": draem,
            "deficit_to_draem_pp": round(draem - hybrid, 2),
            "omitted_count": omitted,
            "omitted_component_fraction": omitted / counts["original_components"],
            "omitted_pixel_fraction": item["omitted_pixel_fraction"],
            "unchanged_count": ghosts,
            "unchanged_component_fraction": ghosts / counts["final_components"],
            "unchanged_pixels": sum(ghost_pixels.values()),
            "ghost_anchor_probability": anchor_probability,
            "expected_ghost_anchors_100_epochs_400_hybrids": anchor_probability * 40000,
        })
    correlations = []
    for feature in ("omitted_count", "omitted_component_fraction", "omitted_pixel_fraction",
                    "unchanged_count", "unchanged_component_fraction"):
        x = np.array([r[feature] for r in rows])
        for outcome in ("external_real_hybrid", "deficit_to_draem_pp"):
            y = np.array([r[outcome] for r in rows])
            leave_one_out = {rows[i]["category"]: float(pearsonr(np.delete(x, i), np.delete(y, i)).statistic)
                             for i in range(len(rows))}
            correlations.append({"feature": feature, "outcome": outcome,
                "n_categories": len(rows), "pearson_r": float(pearsonr(x, y).statistic),
                "spearman_rho": float(spearmanr(x, y).statistic),
                "leave_one_category_out_pearson": leave_one_out})
    result = {"interpretation_of_hybrid_only": "V3 real+hybrid; only synthetic augmentation source is hybrid",
        "gap_definition": "real+DRAEM minus real+hybrid; positive means hybrid performs worse",
        "scope": "Exploratory aggregate correlations across 8 heterogeneous categories; no causal claim",
        "rows": rows, "correlations": correlations}
    (ROOT / "correlations.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
