"""Public entry points for the repository-backed study visualizer."""

from __future__ import annotations

import argparse

from data_handler.visualizer.app import (
    HybridDataGeneratorVisualizer,
    run_hybrid_visualizer,
    run_hybrid_visualizer_for_folder,
)

__all__ = [
    "HybridDataGeneratorVisualizer",
    "run_hybrid_visualizer",
    "run_hybrid_visualizer_for_folder",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse a normalized HybridDataGenerator study."
    )
    parser.add_argument("study_folder")
    parser.add_argument(
        "--channel",
        default="auto",
        help="auto/rgb or a concrete channel index (default: auto)",
    )
    args = parser.parse_args()
    run_hybrid_visualizer_for_folder(args.study_folder, channel=args.channel)


if __name__ == "__main__":
    main()
