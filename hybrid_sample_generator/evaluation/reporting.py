"""Console and histogram reporting for evaluation results."""

import os
from collections import defaultdict

import matplotlib.pyplot as plt


def analyze_results(results):
    if not results or not results["sample_counter"]:
        print("No results.")
        return
    print(f"Analysed {results['sample_counter']} explicit real/synthetic pairs.")
    for name, value in results["mean_real"].items():
        print(f"Real {name}: {value:.4f}")
    for name, value in results["mean_synth"].items():
        print(f"Synthetic {name}: {value:.4f}")


def save_difference_histograms(differences, save_path):
    if not differences:
        return
    columns = 2
    rows = (len(differences) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(10, rows * 4), squeeze=False)
    flat_axes = axes.flatten()
    for axis, (name, values) in zip(flat_axes, differences.items()):
        axis.hist(values, bins=32, edgecolor="black", alpha=0.7)
        axis.set_title(f"Difference Histogram: {name}")
        axis.set_xlabel("Absolute difference")
        axis.set_ylabel("Pairs")
    for axis in flat_axes[len(differences):]:
        axis.set_axis_off()
    figure.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def print_overlap_summary(outlier_groups):
    sample_metrics = defaultdict(set)
    for group in outlier_groups:
        for metric, entries in group.items():
            for entry in entries:
                sample_metrics[entry["sample"]].add(metric)
    if not sample_metrics:
        print("No metric outlier overlaps.")
        return
    counts = defaultdict(int)
    for metrics in sample_metrics.values():
        counts[len(metrics)] += 1
    print(
        "Outlier overlaps: "
        + ", ".join(
            f"{sample_count} samples in {metric_count} metrics"
            for metric_count, sample_count in sorted(counts.items())
        )
    )


__all__ = ["analyze_results", "print_overlap_summary", "save_difference_histograms"]
