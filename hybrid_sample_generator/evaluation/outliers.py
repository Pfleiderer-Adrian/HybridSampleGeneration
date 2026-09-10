"""Metric outlier detection."""

import numpy as np

from hybrid_sample_generator.configuration.evaluation import EvaluationConfiguration


def find_outliers(values, entries, config: EvaluationConfiguration, metric_name):
    if not values:
        return []
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    custom = config.outlier_thresholds.get(metric_name, {})
    lower = custom.get("min") if custom.get("min") is not None else lower
    upper = custom.get("max") if custom.get("max") is not None else upper
    return sorted(
        [entry for entry in entries if entry["value"] < lower or entry["value"] > upper],
        key=lambda entry: abs(entry["value"]),
        reverse=True,
    )


__all__ = ["find_outliers"]
