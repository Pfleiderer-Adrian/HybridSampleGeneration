from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt

from data_handler.visualizer.anomalies_tab import AnomaliesTab
from data_handler.visualizer.evaluation_tab import EvaluationTab
from data_handler.visualizer.hybrids_tab import HybridsTab
from data_handler.visualizer.maintenance import StudyMaintenance
from data_handler.visualizer.overview_tab import OverviewTab
from data_handler.visualizer.queries import StudyBrowserModel
from data_handler.visualizer.relations_tab import RelationsTab
from data_handler.visualizer.rendering import ArrayCache
from data_handler.visualizer.state import SelectionController
from synthesizer.ArtifactStore import ArtifactStore
from synthesizer.StudyPaths import StudyPaths
from synthesizer.StudyRepository import StudyRepository


class HybridDataGeneratorVisualizer:
    """Tabbed master-detail browser for one normalized study repository."""

    TAB_LABELS = (
        "Overview",
        "Anomalies",
        "Hybrids & placements",
        "Evaluation",
        "Data structure",
    )

    def __init__(
        self,
        root,
        repository: StudyRepository,
        artifact_store: ArtifactStore,
        *,
        configuration_path: str | None = None,
        channel: str | int = "auto",
    ) -> None:
        self.root = root
        self.repository = repository
        self.artifact_store = artifact_store
        self.configuration_path = configuration_path
        self.initial_channel = str(channel)
        self.model = StudyBrowserModel(
            repository,
            artifact_store,
            metric_csv_path=str(
                artifact_store.study_folder / "evaluation_results" / "metric_diffs.csv"
            ),
        )
        self.cache = ArrayCache(artifact_store, max_items=18)
        self.selection = SelectionController()
        self.maintenance = StudyMaintenance(self.model)
        self.pages = {}
        self._search_after = None

        root.title("Hybrid Sample Generator – Study Browser")
        root.geometry("1800x1050")
        root.minsize(1100, 700)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, padding=(12, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(
            header,
            text=artifact_store.study_folder.name,
            font=("Arial", 13, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.summary_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.summary_var).grid(
            row=0, column=1, sticky="w", padx=(18, 12)
        )
        self.search_var = tk.StringVar(value="")
        search = ttk.Entry(header, textvariable=self.search_var, width=34)
        search.grid(row=0, column=2, sticky="e", padx=(8, 4))
        search.insert(0, "")
        ttk.Label(header, text="Search IDs / source names").grid(
            row=1, column=2, sticky="e", padx=(8, 4)
        )
        ttk.Button(header, text="Refresh", command=self.refresh_all).grid(
            row=0, column=3, rowspan=2, sticky="ns", padx=(6, 0)
        )

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.holders = []
        for label in self.TAB_LABELS:
            holder = ttk.Frame(self.notebook)
            holder.rowconfigure(0, weight=1)
            holder.columnconfigure(0, weight=1)
            self.notebook.add(holder, text=label)
            self.holders.append(holder)

        footer = ttk.Frame(root, padding=(12, 4))
        footer.grid(row=2, column=0, sticky="ew")
        self.selection_var = tk.StringVar(value="No entity selected")
        ttk.Label(footer, textvariable=self.selection_var).pack(side="left")
        ttk.Label(
            footer,
            text=str(artifact_store.study_folder),
            foreground="#666666",
        ).pack(side="right")

        self.search_var.trace_add("write", self._schedule_search)
        self.notebook.bind("<<NotebookTabChanged>>", self._tab_changed)
        self.selection.subscribe(self._selection_changed)
        root.bind("<Left>", lambda event: self._key_navigation(event, "previous"))
        root.bind("<Right>", lambda event: self._key_navigation(event, "next"))
        root.bind("<Up>", lambda event: self._key_navigation(event, "slice_up"))
        root.bind("<Down>", lambda event: self._key_navigation(event, "slice_down"))

        self._update_summary()
        self._ensure_page(0)

    def _ensure_page(self, index: int):
        if index in self.pages:
            return self.pages[index]
        holder = self.holders[index]
        if index == 0:
            page = OverviewTab(
                holder,
                self.model,
                configuration_path=self.configuration_path,
            )
        elif index == 1:
            page = AnomaliesTab(holder, self.model, self.cache, self.selection)
        elif index == 2:
            page = HybridsTab(holder, self.model, self.cache, self.selection)
        elif index == 3:
            page = EvaluationTab(holder, self.model, self.cache, self.selection)
        else:
            page = RelationsTab(
                holder,
                self.model,
                self.cache,
                self.selection,
                self.maintenance,
                on_data_changed=self.refresh_all,
            )
        page.grid(row=0, column=0, sticky="nsew")
        self.pages[index] = page
        if hasattr(page, "channel_var"):
            page.channel_var.set(self.initial_channel)
            page.render()
        query = self.search_var.get().strip()
        if query:
            page.set_search(query)
        return page

    def _tab_changed(self, _event=None) -> None:
        try:
            index = self.notebook.index(self.notebook.select())
        except tk.TclError:
            return
        page = self._ensure_page(index)
        page.set_search(self.search_var.get())

    def _schedule_search(self, *_args) -> None:
        if self._search_after is not None:
            self.root.after_cancel(self._search_after)
        self._search_after = self.root.after(250, self._apply_search)

    def _apply_search(self) -> None:
        self._search_after = None
        try:
            index = self.notebook.index(self.notebook.select())
        except tk.TclError:
            return
        self._ensure_page(index).set_search(self.search_var.get())

    def refresh_all(self) -> None:
        self.model.refresh()
        self.cache.clear()
        self.selection.clear(source="refresh")
        for page in tuple(self.pages.values()):
            page.refresh()
        self._update_summary()

    def _update_summary(self) -> None:
        summary = self.model.summary()
        self.summary_var.set(
            f"Originals {summary['originals']} · Real {summary['real_anomalies']} · "
            f"Synthetic {summary['synthetic_anomalies']} · Hybrids {summary['hybrids']} · "
            f"Placements {summary['placements']}"
        )

    def _selection_changed(self, state) -> None:
        values = [
            ("O", state.original_sample_id),
            ("R", state.real_anomaly_id),
            ("S", state.synthetic_anomaly_id),
            ("H", state.hybrid_sample_id),
            ("P", state.placement_id),
        ]
        text = " · ".join(
            f"{prefix}:{_short(value)}" for prefix, value in values if value
        )
        self.selection_var.set(text or "No entity selected")

    def _key_navigation(self, event, action: str):
        widget_class = event.widget.winfo_class()
        if widget_class in {"Entry", "TEntry", "Text", "Treeview", "TTreeview"}:
            return None
        index = self.notebook.index(self.notebook.select())
        page = self._ensure_page(index)
        if action == "previous" and hasattr(page, "previous"):
            page.previous()
        elif action == "next" and hasattr(page, "next"):
            page.next()
        elif action == "slice_up" and hasattr(page, "change_slice"):
            page.change_slice(1)
        elif action == "slice_down" and hasattr(page, "change_slice"):
            page.change_slice(-1)
        return "break"

    def close(self) -> None:
        for page in tuple(self.pages.values()):
            page.close()
        plt.close("all")
        self.root.quit()
        self.root.destroy()


def run_hybrid_visualizer(config, channel: str | int = "auto"):
    paths = config.study.paths
    return run_hybrid_visualizer_for_folder(
        paths.study_folder,
        channel=channel,
        configuration_path=paths.configuration_file,
    )


def run_hybrid_visualizer_for_folder(
    study_folder: str,
    channel: str | int = "auto",
    *,
    configuration_path: str | None = None,
):
    folder = Path(study_folder).expanduser().resolve()
    paths = StudyPaths(str(folder), os.path.basename(folder) or "study")
    repository = StudyRepository(paths.artifact_database)
    artifact_store = ArtifactStore(paths.study_folder)
    root = tk.Tk()
    application = HybridDataGeneratorVisualizer(
        root,
        repository,
        artifact_store,
        configuration_path=configuration_path or paths.configuration_file,
        channel=channel,
    )
    root.mainloop()
    return application


def _short(value: str) -> str:
    return value if len(value) <= 12 else value[-10:]
