import json
import math
import argparse
from typing import Any, Dict, List, Optional


def _ceil_to_decimals(x: float, decimals: int = 3) -> float:
    """Round UP to a fixed number of decimal places."""
    factor = 10 ** decimals
    return math.ceil(x * factor) / factor


def _safe_div(n: float, d: float) -> Optional[float]:
    """Safe division returning None if undefined."""
    if d == 0:
        return None
    return n / d


def _extract_metric_per_case(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and "metric_per_case" in obj:
        if not isinstance(obj["metric_per_case"], list):
            raise ValueError('"metric_per_case" must be a list.')
        return obj["metric_per_case"]
    if isinstance(obj, list):
        return obj
    raise ValueError('Input must be a JSON dict with key "metric_per_case" or a list of cases.')


def _extract_mean_block(obj: Any, class_key: str) -> Dict[str, float]:
    """
    nnU-Net summaries often have:
      - "foreground_mean": {...}
      - or "mean": {"1": {...}}
    Prefer mean["<class_key>"] if present, else foreground_mean.

    Returns FRACTIONS (0..1) for metrics and ratio for Area Ratio.
    """
    if not isinstance(obj, dict):
        raise ValueError("To use --mode mean, the JSON must be an object containing mean/foreground_mean blocks.")

    block = None
    if "mean" in obj and isinstance(obj["mean"], dict) and class_key in obj["mean"]:
        block = obj["mean"][class_key]
    elif "foreground_mean" in obj and isinstance(obj["foreground_mean"], dict):
        block = obj["foreground_mean"]

    if not isinstance(block, dict):
        raise ValueError(f"Could not find mean['{class_key}'] or 'foreground_mean' in JSON.")

    dice = float(block.get("Dice", float("nan")))
    iou = float(block.get("IoU", float("nan")))

    tp = float(block.get("TP", 0.0))
    tn = float(block.get("TN", 0.0))
    fp = float(block.get("FP", 0.0))
    fn = float(block.get("FN", 0.0))

    tnr = _safe_div(tn, tn + fp)
    tpr = _safe_div(tp, tp + fn)
    fnr = None if tpr is None else 1.0 - tpr
    fpr = None if tnr is None else 1.0 - tnr

    n_pred = float(block.get("n_pred", tp + fp))
    n_ref = float(block.get("n_ref", tp + fn))
    area_ratio = _safe_div(n_pred, n_ref)

    return {
        "DICE": dice,
        "IoU": iou,
        "TNR": float("nan") if tnr is None else tnr,
        "TPR": float("nan") if tpr is None else tpr,
        "FNR": float("nan") if fnr is None else fnr,
        "FPR": float("nan") if fpr is None else fpr,
        "Area Ratio": float("nan") if area_ratio is None else area_ratio,
    }


def _macro_means_from_metric_per_case(
    metric_per_case: List[Dict[str, Any]],
    class_key: str = "1",
) -> Dict[str, float]:
    """
    Returns MACRO (case-wise mean) metrics as FRACTIONS (0..1), not percent:
      DICE, IoU, TNR, TPR, FNR, FPR, Area Ratio (= n_pred / n_ref).
    """
    dice_vals: List[float] = []
    iou_vals: List[float] = []
    tnr_vals: List[float] = []
    tpr_vals: List[float] = []
    fnr_vals: List[float] = []
    fpr_vals: List[float] = []
    ar_vals: List[float] = []

    for item in metric_per_case:
        m = item.get("metrics", {}).get(class_key, {})
        if not m:
            continue

        tp = float(m.get("TP", 0.0))
        tn = float(m.get("TN", 0.0))
        fp = float(m.get("FP", 0.0))
        fn = float(m.get("FN", 0.0))

        # Dice / IoU (prefer provided)
        if "Dice" in m:
            dice_vals.append(float(m["Dice"]))
        else:
            d = _safe_div(2 * tp, (2 * tp + fp + fn))
            if d is not None:
                dice_vals.append(d)

        if "IoU" in m:
            iou_vals.append(float(m["IoU"]))
        else:
            j = _safe_div(tp, (tp + fp + fn))
            if j is not None:
                iou_vals.append(j)

        # Rates (per case)
        tnr = _safe_div(tn, tn + fp)  # specificity
        tpr = _safe_div(tp, tp + fn)  # sensitivity/recall

        if tnr is not None:
            tnr_vals.append(tnr)
            fpr_vals.append(1.0 - tnr)

        if tpr is not None:
            tpr_vals.append(tpr)
            fnr_vals.append(1.0 - tpr)

        # Area Ratio per case (n_pred / n_ref)
        n_pred = float(m.get("n_pred", tp + fp))
        n_ref = float(m.get("n_ref", tp + fn))
        ar = _safe_div(n_pred, n_ref)
        if ar is not None:
            ar_vals.append(ar)

    def mean(xs: List[float]) -> float:
        return float("nan") if not xs else sum(xs) / len(xs)

    return {
        "DICE": mean(dice_vals),
        "IoU": mean(iou_vals),
        "TNR": mean(tnr_vals),
        "TPR": mean(tpr_vals),
        "FNR": mean(fnr_vals),
        "FPR": mean(fpr_vals),
        "Area Ratio": mean(ar_vals),
    }


def _compute_metrics(obj: Any, class_key: str, mode: str) -> Dict[str, float]:
    """
    mode:
      - 'macro': compute case-wise mean from metric_per_case
      - 'mean' : read precomputed mean/foreground_mean block from summary JSON
    Returns fractions (0..1) for all metrics and ratio for Area Ratio.
    """
    mode = mode.lower()
    if mode == "macro":
        mpc = _extract_metric_per_case(obj)
        return _macro_means_from_metric_per_case(mpc, class_key=class_key)
    if mode == "mean":
        return _extract_mean_block(obj, class_key=class_key)
    raise ValueError("mode must be either 'macro' or 'mean'.")


def print_metrics_with_baseline_abs_delta(
    new_obj: Any,
    base_obj: Any,
    class_key: str = "1",
    mode: str = "macro",
) -> None:
    """
    Prints (rounded UP to 3 decimals):
      METRIC: <new_value%>  (Δ vs baseline: <abs_delta%>)
    where abs_delta is (new - baseline) in absolute percentage points
    (or absolute percent units for Area Ratio since we also display it as %).
    """
    order = ["DICE", "IoU", "TNR", "TPR", "FNR", "FPR", "Area Ratio"]

    new_vals = _compute_metrics(new_obj, class_key=class_key, mode=mode)
    base_vals = _compute_metrics(base_obj, class_key=class_key, mode=mode)

    for k in order:
        n = new_vals[k]
        b = base_vals[k]

        n_pct = n * 100.0
        b_pct = b * 100.0
        delta_pct = (n_pct - b_pct)

        # Round UP to 3 decimals
        n_pct_up = _ceil_to_decimals(n_pct, 3) if not math.isnan(n_pct) else n_pct

        if math.isnan(delta_pct):
            delta_str = "n/a"
        else:
            # absolute difference can be negative; we still round "up" numerically:
            # e.g. -0.1234 -> -0.123 (ceil), which is closer to 0.
            delta_up = _ceil_to_decimals(delta_pct, 3)
            sign = "+" if delta_up >= 0 else ""
            delta_str = f"{sign}{delta_up:.3f}"

        print(f"{k}: {n_pct_up:.3f}% ({delta_str})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute segmentation metrics and ABSOLUTE difference vs baseline.\n"
            "Modes:\n"
            "  macro: compute case-wise mean from metric_per_case\n"
            "  mean : read precomputed mean/foreground_mean block from summary JSON\n"
            "Outputs are percent values rounded UP to 3 decimals, and Δ is absolute percentage points."
        )
    )
    parser.add_argument("new_json_path", help="Path to JSON for the NEW run.")
    parser.add_argument("baseline_json_path", help="Path to JSON for the BASELINE run.")
    parser.add_argument("--class-key", default="1", help="Class key inside each case's 'metrics' dict (default: '1').")
    parser.add_argument(
        "--mode",
        choices=["macro", "mean"],
        default="macro",
        help="Choose 'macro' (from metric_per_case) or 'mean' (from summary mean block). Default: macro.",
    )
    args = parser.parse_args()

    with open(args.new_json_path, "r", encoding="utf-8") as f:
        new_obj = json.load(f)
    with open(args.baseline_json_path, "r", encoding="utf-8") as f:
        base_obj = json.load(f)

    print_metrics_with_baseline_abs_delta(
        new_obj=new_obj,
        base_obj=base_obj,
        class_key=args.class_key,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
