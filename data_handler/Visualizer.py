from __future__ import annotations

import argparse
import os
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.StudyPaths import StudyPaths
from synthesizer.StudyRepository import StudyRepository


def build_study_hierarchy(repository: StudyRepository) -> list[dict]:
    """Return Original → Hybrid → Placement → Synthetic → Real as typed records."""
    originals = OrderedDict(
        (
            original.id,
            {"original": original, "hybrids": OrderedDict()},
        )
        for original in repository.list_original_samples()
    )
    for hybrid in repository.list_hybrid_samples():
        parent = originals.get(hybrid.original_sample_id)
        if parent is not None:
            parent["hybrids"][hybrid.id] = {
                "hybrid": hybrid,
                "placements": [],
            }
    for entry in repository.hierarchy():
        hybrid_node = originals[entry.original.id]["hybrids"][entry.hybrid.id]
        hybrid_node["placements"].append(
            {
                "placement": entry.placement,
                "synthetic_anomaly": entry.synthetic_anomaly,
                "real_anomaly": entry.real_anomaly,
            }
        )
    return [
        {**node, "hybrids": list(node["hybrids"].values())}
        for node in originals.values()
    ]


class HybridDataGeneratorVisualizer:
    """Small database-backed browser for generated study relationships."""

    def __init__(
        self,
        root,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        *,
        channel: int = 0,
        cmap: str = "gray",
    ) -> None:
        self.root = root
        self.repository = repository
        self.artifact_store = artifact_store
        self.channel = int(channel)
        self.cmap = cmap
        self._items = {}

        root.title("Hybrid Sample Generator – Study Browser")
        root.geometry("1500x900")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        side = ttk.Frame(root, padding=8)
        side.grid(row=0, column=0, sticky="nsew")
        side.rowconfigure(1, weight=1)
        ttk.Label(
            side,
            text="Original → Hybrid → Placement → Synthetic → Real",
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.tree = ttk.Treeview(side, show="tree", height=35)
        self.tree.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(side, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        content = ttk.Frame(root, padding=8)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        self.axes = axes.flatten()
        self.canvas = FigureCanvasTkAgg(self.figure, master=content)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.status = tk.StringVar(value="")
        ttk.Label(content, textvariable=self.status).grid(row=1, column=0, sticky="w")

        self.refresh()

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._items.clear()
        for original_node in build_study_hierarchy(self.repository):
            original = original_node["original"]
            original_item = self.tree.insert(
                "", "end", text=f"Original: {original.source_name} [{original.id}]", open=True
            )
            self._items[original_item] = {"original": original}
            for hybrid_node in original_node["hybrids"]:
                hybrid = hybrid_node["hybrid"]
                hybrid_item = self.tree.insert(
                    original_item,
                    "end",
                    text=f"Hybrid {hybrid.variant_index}: {hybrid.status} [{hybrid.id}]",
                    open=True,
                )
                self._items[hybrid_item] = {"original": original, "hybrid": hybrid}
                for placement_node in hybrid_node["placements"]:
                    placement = placement_node["placement"]
                    synthetic = placement_node["synthetic_anomaly"]
                    real = placement_node["real_anomaly"]
                    placement_item = self.tree.insert(
                        hybrid_item,
                        "end",
                        text=(
                            f"Placement {placement.order_index}: {placement.position} "
                            f"[{placement.id}]"
                        ),
                    )
                    payload = {
                        "original": original,
                        "hybrid": hybrid,
                        "placement": placement,
                        "synthetic": synthetic,
                        "real": real,
                    }
                    self._items[placement_item] = payload
                    synthetic_item = self.tree.insert(
                        placement_item,
                        "end",
                        text=f"Synthetic variant {synthetic.variant_index} [{synthetic.id}]",
                    )
                    self._items[synthetic_item] = payload
                    real_item = self.tree.insert(
                        synthetic_item,
                        "end",
                        text=f"Real component {real.component_index} [{real.id}]",
                    )
                    self._items[real_item] = payload
        roots = self.tree.get_children()
        if roots:
            self.tree.selection_set(roots[0])
            self._on_select()

    def _on_select(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        payload = self._items[selected[0]]
        original = payload.get("original")
        hybrid = payload.get("hybrid")
        placement = payload.get("placement")
        synthetic = payload.get("synthetic")
        real = payload.get("real")
        panels = (
            (
                "Original sample",
                original.image_path if original else None,
                original.segmentation_path if original else None,
            ),
            (
                "Hybrid sample",
                hybrid.image_path if hybrid else None,
                hybrid.segmentation_path if hybrid else None,
            ),
            (
                "Placement ROI",
                placement.roi_image_path if placement else None,
                placement.roi_segmentation_path if placement else None,
            ),
            (
                "Synthetic anomaly",
                synthetic.image_path if synthetic else None,
                synthetic.segmentation_path if synthetic else None,
            ),
            (
                "Real anomaly",
                real.image_path if real else None,
                real.segmentation_path if real else None,
            ),
            (
                "Real source ROI",
                real.roi_image_path if real else None,
                real.roi_segmentation_path if real else None,
            ),
        )
        for axis, (title, image_path, mask_path) in zip(self.axes, panels):
            axis.clear()
            axis.set_axis_off()
            axis.set_title(title)
            if not image_path or not self.artifact_store.exists(image_path):
                axis.text(0.5, 0.5, "not available", ha="center", va="center")
                continue
            image = _display_plane(
                self.artifact_store.load_array(image_path), channel=self.channel
            )
            axis.imshow(image, cmap=self.cmap)
            if mask_path and self.artifact_store.exists(mask_path):
                mask = _display_plane(self.artifact_store.load_array(mask_path), channel=0)
                axis.imshow(np.ma.masked_where(mask <= 0, mask), cmap="autumn", alpha=0.35)
        self.status.set(
            f"{len(self.repository.list_original_samples())} originals, "
            f"{len(self.repository.list_hybrid_samples())} hybrids, "
            f"{len(self.repository.list_placements())} placements"
        )
        self.canvas.draw_idle()

    def close(self) -> None:
        plt.close(self.figure)
        self.root.quit()
        self.root.destroy()


def _display_plane(array: np.ndarray, *, channel: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 4:
        channel = min(max(channel, 0), array.shape[0] - 1)
        return array[channel, array.shape[1] // 2]
    if array.ndim == 3:
        channel = min(max(channel, 0), array.shape[0] - 1)
        return array[channel]
    if array.ndim == 2:
        return array
    raise ValueError(f"Cannot visualize array with shape {array.shape}.")


def run_hybrid_visualizer(config, channel: int = 0, cmap: str = "gray"):
    paths = config.study.paths
    return run_hybrid_visualizer_for_folder(
        paths.study_folder, channel=channel, cmap=cmap
    )


def run_hybrid_visualizer_for_folder(
    study_folder: str, channel: int = 0, cmap: str = "gray"
):
    study_folder = str(Path(study_folder).expanduser().resolve())
    paths = StudyPaths(study_folder, os.path.basename(study_folder) or "study")
    repository = StudyRepository(paths.artifact_database)
    artifact_store = ArtifactStore(paths.study_folder)
    root = tk.Tk()
    root.tk.call("tk", "scaling", 2.0)
    HybridDataGeneratorVisualizer(
        root, repository, artifact_store, channel=channel, cmap=cmap
    )
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Browse one hybrid-generation study.")
    parser.add_argument("study_folder")
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()
    run_hybrid_visualizer_for_folder(args.study_folder, channel=args.channel)


if __name__ == "__main__":
    main()
