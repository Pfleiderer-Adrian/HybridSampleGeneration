"""Intensity-relation analysis and context normalization for classical fusion."""

import numpy as np
from scipy.ndimage import binary_dilation

def infer_output_intensity_bounds(values, eps=1e-6):
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    mn = float(np.min(finite))
    mx = float(np.max(finite))
    if mn >= -eps and mx <= 1.0 + eps:
        return 0.0, 1.0
    if mn >= -eps and mx <= 255.0 + eps:
        return 0.0, 255.0
    return None


def _fit_values_into_bounds(values, bounds, eps=1e-8):
    if bounds is None:
        return values

    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values

    lo, hi = bounds
    current_min = float(np.min(finite))
    current_max = float(np.max(finite))
    if current_min >= lo and current_max <= hi:
        return values

    # simple shift if the value spread already fits into the bounds
    value_span = current_max - current_min
    bound_span = hi - lo
    if value_span <= bound_span + eps:
        offset = 0.0
        if current_max > hi:
            offset = hi - current_max
        if current_min + offset < lo:
            offset = lo - current_min
        return np.clip(values + offset, lo, hi)

    # spread too large => compress around the source and target centers
    value_center = 0.5 * (current_min + current_max)
    bound_center = 0.5 * (lo + hi)
    scale = bound_span / max(value_span, eps)
    fitted = bound_center + (values - value_center) * scale
    return np.clip(fitted, lo, hi)


def _region_stats(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    q25, q50, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    return {
        "median": float(q50),
        "iqr": max(float(q75 - q25), float(eps)),
    }


def _relation_for_mask(roi, anomaly_mask, context_mask, relation_mode, eps=1e-8):
    """Measure one anomaly/context relation jointly across all channels."""
    anomaly_stats = _region_stats(roi[:, anomaly_mask], eps=eps)
    context_stats = _region_stats(roi[:, context_mask], eps=eps)
    if anomaly_stats is None or context_stats is None:
        return None

    relation = {
        "iqr_ratio": float(anomaly_stats["iqr"] / max(context_stats["iqr"], eps)),
    }
    context_median = context_stats["median"]
    if relation_mode == "delta":
        relation["median_delta"] = float(anomaly_stats["median"] - context_median)
    elif relation_mode == "ratio":
        relation["median_ratio"] = (
            float(anomaly_stats["median"] / context_median)
            if abs(context_median) > eps
            else None
        )
    else:
        raise ValueError(
            "fusion_relation_mode must be 'delta' or 'ratio'. "
            f"Got {relation_mode!r}."
        )
    return relation


def anomaly_context_relations(
    roi,
    roi_mask,
    border_width,
    relation_mode,
    eps=1e-8,
    min_context_size=8,
    norm_classes_separately=False,
):
    if roi is None or roi_mask is None:
        return None

    roi = np.asarray(roi, dtype=np.float32)
    roi_mask = np.asarray(roi_mask)
    if roi.ndim < 3 or roi_mask.ndim != roi.ndim:
        return None

    label_mask = np.max(roi_mask, axis=0)
    spatial_mask = label_mask > 0
    if not np.any(spatial_mask):
        return None

    spatial_ndim = roi.ndim - 1
    kernel_size = max(int(border_width), 1) * 2 + 1
    structure = np.ones((kernel_size,) * spatial_ndim, dtype=bool)

    if not norm_classes_separately:
        dilated_mask = binary_dilation(spatial_mask, structure=structure)
        context_mask = dilated_mask & ~spatial_mask
        if np.count_nonzero(context_mask) < int(min_context_size):
            context_mask = ~spatial_mask
        if np.count_nonzero(context_mask) < int(min_context_size):
            return None

        return _relation_for_mask(roi, spatial_mask, context_mask, relation_mode, eps=eps)

    relations_by_label = {}
    global_context_mask = None

    for label_value in np.unique(label_mask):
        if label_value <= 0:
            continue
        class_mask = label_mask == label_value
        dilated_mask = binary_dilation(class_mask, structure=structure)
        context_mask = dilated_mask & ~spatial_mask
        if np.count_nonzero(context_mask) < int(min_context_size):
            if global_context_mask is None:
                global_context_mask = ~spatial_mask
            context_mask = global_context_mask
        if np.count_nonzero(context_mask) < int(min_context_size):
            continue

        relations_by_label[int(label_value)] = _relation_for_mask(
            roi, class_mask, context_mask, relation_mode, eps=eps
        )

    return relations_by_label or None


def _target_median_from_relation(bg_median, relation, relation_mode, eps=1e-8):
    if not relation:
        return bg_median

    if relation_mode == "delta":
        delta = relation.get("median_delta")
        if delta is not None and np.isfinite(delta):
            return float(bg_median + float(delta))
        return bg_median

    if relation_mode == "ratio":
        ratio = relation.get("median_ratio")
        if ratio is not None and np.isfinite(ratio) and abs(bg_median) > eps:
            return float(bg_median * float(ratio))
        return bg_median

    raise ValueError(f"fusion_relation_mode must be 'delta' or 'ratio'. Got {relation_mode!r}.")


def normalize_anomaly_to_context(
    anom,
    context_slice,
    blend_bg_slice,
    anomaly_mask,
    context_mask,
    original_relations,
    relation_mode,
    class_label=None,
    alpha_mask=None,
    eps=1e-8,
    output_intensity_bounds=None,
):
    matched = anom.copy()
    alpha_eff = None
    if alpha_mask is not None:
        alpha_values = np.asarray(alpha_mask, dtype=np.float32)[anomaly_mask]
        alpha_values = alpha_values[np.isfinite(alpha_values) & (alpha_values > eps)]
        if alpha_values.size > 0:
            alpha_eff = float(np.clip(np.max(alpha_values), eps, 1.0))

    anomaly_values = matched[:, anomaly_mask]
    context_values = context_slice[:, context_mask]
    anomaly_stats = _region_stats(anomaly_values, eps=eps)
    context_stats = _region_stats(context_values, eps=eps)
    if anomaly_stats is None or context_stats is None:
        return matched

    relations = original_relations
    if class_label is not None and isinstance(relations, dict):
        relations = relations.get(int(class_label))
    relation = relations if isinstance(relations, dict) else None

    target_median = _target_median_from_relation(
        context_stats["median"], relation, relation_mode, eps=eps
    )
    target_iqr = context_stats["iqr"]
    if relation is not None:
        iqr_ratio = relation.get("iqr_ratio")
        if iqr_ratio is not None and np.isfinite(iqr_ratio):
            target_iqr = max(float(context_stats["iqr"] * float(iqr_ratio)), float(eps))

    pre_target_median = target_median
    pre_target_iqr = target_iqr
    if alpha_eff is not None and alpha_eff < 1.0:
        inside_bg_stats = _region_stats(blend_bg_slice[:, anomaly_mask], eps=eps)
        if inside_bg_stats is not None:
            pre_target_median = (
                target_median - inside_bg_stats["median"] * (1.0 - alpha_eff)
            ) / alpha_eff
            pre_target_iqr = max(float(target_iqr / alpha_eff), float(eps))

    joint_values = (
        (anomaly_values - anomaly_stats["median"]) / anomaly_stats["iqr"]
    ) * pre_target_iqr + pre_target_median
    joint_values = _fit_values_into_bounds(
        joint_values, output_intensity_bounds, eps=eps
    )
    matched[:, anomaly_mask] = joint_values
    return matched


def match_local_intensity(
    anom,
    ctrl,
    bg_slice,
    valid_mask,
    target_mask,
    anomaly_roi,
    anomaly_roi_mask,
    alpha_mask,
    params,
    normalization_eps=1e-8,
):
    """Match anomaly intensity to local context and preserve ROI relations."""
    binary_mask = valid_mask > 0
    normalization_border_width = getattr(
        params, "fusion_normalization_border_width", 2
    )
    if normalization_border_width is None or not np.any(binary_mask):
        return anom

    border_width = int(normalization_border_width)
    original_relations = None
    eps = float(normalization_eps)
    output_intensity_bounds = infer_output_intensity_bounds(ctrl)
    min_context_size = int(getattr(params, "fusion_relation_min_context_size", 8))
    relation_mode = getattr(params, "fusion_relation_mode", "delta")
    norm_classes_separately = bool(
        getattr(params, "fusion_relation_norm_classes_separately", False)
    )

    if border_width == -1:
        context_slice = ctrl
        context_mask = np.ones(ctrl.shape[1:], dtype=bool)
        fallback_context_mask = context_mask
        dilation_structure = None
    elif border_width >= 0:
        dilation_kernel_size = border_width * 2 + 1
        dilation_structure = np.ones(
            (dilation_kernel_size,) * binary_mask.ndim, dtype=bool
        )
        context_slice = bg_slice
        fallback_context_mask = ~binary_mask
        dilated_mask = binary_dilation(binary_mask, structure=dilation_structure)
        context_mask = dilated_mask & fallback_context_mask
        if np.count_nonzero(context_mask) < min_context_size:
            context_mask = fallback_context_mask
        if getattr(params, "fusion_restore_anomaly_bg_relation", None):
            original_relations = anomaly_context_relations(
                anomaly_roi,
                anomaly_roi_mask,
                border_width,
                relation_mode,
                eps=eps,
                min_context_size=min_context_size,
                norm_classes_separately=norm_classes_separately,
            )
    else:
        raise ValueError(
            "fusion_normalization_border_width must be None, -1, or >= 0."
        )

    labels = np.unique(target_mask[binary_mask])
    labels = labels[labels > 0]

    if not norm_classes_separately:
        if np.count_nonzero(context_mask) < min_context_size:
            return anom
        return normalize_anomaly_to_context(
            anom,
            context_slice,
            bg_slice,
            binary_mask,
            context_mask,
            original_relations,
            relation_mode,
            class_label=None,
            alpha_mask=alpha_mask,
            eps=eps,
            output_intensity_bounds=output_intensity_bounds,
        )

    matched = anom
    for label_value in labels:
        class_mask = target_mask == label_value
        if not np.any(class_mask):
            continue
        if border_width == -1:
            class_context_mask = context_mask
        else:
            class_dilated_mask = binary_dilation(
                class_mask, structure=dilation_structure
            )
            class_context_mask = class_dilated_mask & ~binary_mask
            if np.count_nonzero(class_context_mask) < min_context_size:
                class_context_mask = context_mask
            if np.count_nonzero(class_context_mask) < min_context_size:
                class_context_mask = fallback_context_mask
        if np.count_nonzero(class_context_mask) < min_context_size:
            continue
        matched = normalize_anomaly_to_context(
            matched,
            context_slice,
            bg_slice,
            class_mask,
            class_context_mask,
            original_relations,
            relation_mode,
            class_label=label_value,
            alpha_mask=alpha_mask,
            eps=eps,
            output_intensity_bounds=output_intensity_bounds,
        )

    return matched


__all__ = [
    "anomaly_context_relations",
    "infer_output_intensity_bounds",
    "match_local_intensity",
    "normalize_anomaly_to_context",
]
