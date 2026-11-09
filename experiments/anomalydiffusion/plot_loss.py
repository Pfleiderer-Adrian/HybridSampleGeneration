"""Plot logged AD batch losses and robust trends without touching running jobs."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse(path, phase):
    pattern = rf"{phase} step=(\d+)/(\d+) loss=([0-9.eE+-]+)"
    return [(int(step), float(value)) for step, total, value in
            re.findall(pattern, path.read_text(errors="replace"))]


def rolling(values, window=51):
    result = np.full((len(values), 3), np.nan)
    for i in range(9, len(values)):
        result[i] = np.quantile(values[max(0, i-window+1):i+1], [.25, .5, .75])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(ZoneInfo("Europe/Berlin")).isoformat()
    summary = {"generated_at": timestamp, "source_launch": str(args.launch.resolve()),
               "sampling": "One logged training batch every 100 steps; rounded to seven decimals",
               "smoothing": "Trailing median and interquartile range of 51 logged batches (~5100 steps)",
               "caution": "Training loss only, no validation-loss or downstream-quality measurement", "phases": {}}
    for phase, stage, label in (("appearance", "appearance_training", "Anomalietraining"),
                                ("mask", "mask_training", "Maskentraining")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        summary["phases"][phase] = {}
        for ax, category in zip(axes, ("can", "fabric")):
            path = args.launch / "jobs" / category / f"{stage}.log"
            pairs = parse(path, phase)
            steps, values = np.array([s for s, v in pairs]), np.array([v for s, v in pairs])
            if not len(steps) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid or missing logged loss data: {path}")
            bands = rolling(values)
            plotted = np.maximum(values, 5e-8)
            ax.scatter(steps / 1000, plotted, s=7, color="#8e9bad", alpha=.27,
                       label="Protokollierter Batch-Loss", rasterized=True)
            ax.fill_between(steps / 1000, np.maximum(bands[:, 0], 5e-8),
                            np.maximum(bands[:, 2], 5e-8), color="#2867a0", alpha=.14,
                            label="Rollierendes 25.–75. Perzentil")
            ax.plot(steps / 1000, np.maximum(bands[:, 1], 5e-8), color="#12558a", linewidth=2.1,
                    label="Rollierender Median")
            ax.set_yscale("log")
            ax.set_title(f"{category} · {steps[-1]:,} Schritte".replace(",", "."))
            ax.set_xlabel("Trainingsschritte (Tausend)")
            ax.set_ylabel("Trainingsverlust · logarithmische Skala")
            ax.grid(True, alpha=.2)
            ax.set_xlim(left=0)
            ax.legend(fontsize=8, loc="best")
            windows = []
            for lo in range(0, int(steps[-1]), 10000):
                selected = values[(steps > lo) & (steps <= lo+10000)]
                if len(selected):
                    q25, median, q75 = np.quantile(selected, [.25, .5, .75])
                    windows.append({"start_step_exclusive": lo, "end_step_inclusive": min(lo+10000, int(steps[-1])),
                                    "complete_10k_window": bool(lo+10000 <= steps[-1]), "logged_batches": len(selected),
                                    "median_loss": float(median), "q25": float(q25), "q75": float(q75)})
            summary["phases"][phase][category] = {"latest_step": int(steps[-1]),
                "logged_batches": len(values), "latest_loss": float(values[-1]),
                "zero_rounded_values": int((values == 0).sum()), "windows": windows}
            with (args.output / f"{category}_{phase}_loss.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("step", "logged_batch_loss", "rolling_q25", "rolling_median", "rolling_q75"))
                writer.writerows((s, v, *band) for (s, v), band in zip(pairs, bands))
        fig.suptitle(f"AnomalyDiffusion · {label} · Stand {timestamp[:16].replace('T', ' ')} MEZ", fontsize=12)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(args.output / f"{phase}_loss.{extension}", dpi=180)
        plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({phase: {cat: {"latest_step": row["latest_step"], "last_window": row["windows"][-1]}
        for cat, row in values.items()} for phase, values in summary["phases"].items()}, indent=2))


if __name__ == "__main__":
    main()
