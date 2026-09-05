from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_handler.visualizer.common import EntityBrowserTab, insert_tree_scrollbars
from data_handler.visualizer.queries import (
    EvaluationGroup,
    StudyBrowserModel,
    filter_evaluation_groups,
)
from data_handler.visualizer.rendering import ArrayCache, PanelSpec
from data_handler.visualizer.state import SelectionController


class EvaluationTab(EntityBrowserTab):
    def __init__(
        self,
        master,
        model: StudyBrowserModel,
        cache: ArrayCache,
        selection: SelectionController,
    ) -> None:
        self.model = model
        self.selection = selection
        self.group: EvaluationGroup | None = None
        self.filtered_groups: list[EvaluationGroup] = []
        self._item_group = {}
        self._items = []
        self._metric_vars = {}
        super().__init__(master, cache=cache, rows=2, columns=2)

        ttk.Label(
            self.side,
            text="Evaluation and outliers",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")
        scope_row = ttk.Frame(self.side)
        scope_row.pack(fill="x", pady=(6, 0))
        ttk.Label(scope_row, text="Scope").pack(side="left")
        self.scope_var = tk.StringVar(value="all")
        scope = ttk.Combobox(
            scope_row,
            textvariable=self.scope_var,
            values=("all", "cutout", "placement"),
            state="readonly",
            width=11,
        )
        scope.pack(side="right")
        scope.bind("<<ComboboxSelected>>", lambda _event: self.refresh())

        self.top_percent_var = tk.DoubleVar(value=1.0)
        self.top_percent_label = tk.StringVar(value="Outlier threshold: top 1.0%")
        ttk.Label(self.side, textvariable=self.top_percent_label).pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Scale(
            self.side,
            from_=0.1,
            to=100.0,
            variable=self.top_percent_var,
            command=self._threshold_changed,
        ).pack(fill="x")

        ttk.Label(self.side, text="Metrics").pack(anchor="w", pady=(8, 0))
        self.metric_frame = ttk.Frame(self.side)
        self.metric_frame.pack(fill="x")
        self._rebuild_metric_controls()

        self.tree = insert_tree_scrollbars(
            self.side,
            columns=("scope", "score"),
            headings=("Scope", "Score"),
            height=13,
        )
        self.tree.heading("#0", text="Real → Synthetic")
        self.tree.column("#0", width=200, stretch=True)
        self.tree.column("scope", width=70, stretch=False)
        self.tree.column("score", width=55, stretch=False)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.build_display_controls()

        self.content.rowconfigure(0, weight=3)
        self.content.rowconfigure(1, weight=1)
        self.histogram_figure, self.histogram_axis = plt.subplots(
            1, 1, figsize=(10, 2.5), constrained_layout=True
        )
        self.histogram_canvas = FigureCanvasTkAgg(
            self.histogram_figure, master=self.content
        )
        self.histogram_canvas.get_tk_widget().grid(
            row=1, column=0, sticky="nsew", pady=(4, 0)
        )
        self.refresh()

    def refresh(self) -> None:
        self._rebuild_metric_controls()
        selected_key = self.group.key if self.group else None
        active_metrics = tuple(
            name for name, variable in self._metric_vars.items() if variable.get()
        )
        self.filtered_groups = filter_evaluation_groups(
            self.model.evaluations,
            metrics=active_metrics,
            top_percent=self.top_percent_var.get(),
            scope=self.scope_var.get(),
            query=self.search_query,
        )
        self.tree.delete(*self.tree.get_children())
        self._item_group.clear()
        self._items.clear()
        preferred = None
        for group in self.filtered_groups:
            label = f"{_short(group.real_anomaly_id)} → {_short(group.synthetic_anomaly_id)}"
            item = self.tree.insert(
                "",
                "end",
                text=label,
                values=(group.scope, f"{group.score:.3f}" if active_metrics else "–"),
            )
            self._item_group[item] = group
            self._items.append(item)
            if group.key == selected_key:
                preferred = item
        target = preferred or (self._items[0] if self._items else None)
        if target:
            self.tree.selection_set(target)
            self.tree.focus(target)
            self.tree.see(target)
            self._on_select()
        else:
            self.group = None
            self.image_grid.set_specs(())
            self.set_details(
                (
                    "Evaluation",
                    {
                        "message": "No evaluation rows match the current filters.",
                        "csv": self.model.metric_csv_path,
                    },
                )
            )
            self._render_histogram()

    def _rebuild_metric_controls(self) -> None:
        metric_names = sorted(
            {name for group in self.model.evaluations for name in group.metrics}
        )
        if metric_names == list(self._metric_vars):
            return
        previous = {name: variable.get() for name, variable in self._metric_vars.items()}
        for widget in self.metric_frame.winfo_children():
            widget.destroy()
        self._metric_vars = {}
        for index, name in enumerate(metric_names):
            variable = tk.BooleanVar(value=previous.get(name, False))
            self._metric_vars[name] = variable
            ttk.Checkbutton(
                self.metric_frame,
                text=name,
                variable=variable,
                command=self.refresh,
            ).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 4))

    def _threshold_changed(self, _value=None) -> None:
        value = float(self.top_percent_var.get())
        self.top_percent_label.set(f"Outlier threshold: top {value:.1f}%")
        if any(variable.get() for variable in self._metric_vars.values()):
            self.refresh()

    def _on_select(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        group = self._item_group[selected[0]]
        self.group = group
        self.slice_var.set(0)
        placement = (
            self.model.placement_by_id.get(group.placement_id)
            if group.placement_id
            else None
        )
        hybrid_id = placement.hybrid_sample_id if placement else None
        self.selection.update(
            source="evaluation",
            real_anomaly_id=group.real_anomaly_id,
            synthetic_anomaly_id=group.synthetic_anomaly_id,
            placement_id=group.placement_id,
            hybrid_sample_id=hybrid_id,
            original_sample_id=(
                self.model.hybrid_by_id[hybrid_id].original_sample_id
                if hybrid_id in self.model.hybrid_by_id
                else None
            ),
        )
        self._show_group()

    def _show_group(self) -> None:
        if self.group is None:
            return
        real = self.model.real_by_id.get(self.group.real_anomaly_id)
        synthetic = self.model.synthetic_by_id.get(self.group.synthetic_anomaly_id)
        placement = (
            self.model.placement_by_id.get(self.group.placement_id)
            if self.group.placement_id
            else None
        )
        self.image_grid.set_specs(
            (
                PanelSpec(
                    "Real anomaly",
                    real.image_path if real else None,
                    real.segmentation_path if real else None,
                    reference_path=(real.roi_image_path if real else None),
                ),
                PanelSpec(
                    "Synthetic anomaly",
                    synthetic.image_path if synthetic else None,
                    synthetic.segmentation_path if synthetic else None,
                    reference_path=(real.roi_image_path if real else None),
                ),
                PanelSpec(
                    "Real source ROI",
                    real.roi_image_path if real else None,
                    real.roi_segmentation_path if real else None,
                ),
                PanelSpec(
                    "Fused placement ROI",
                    placement.roi_image_path if placement else None,
                    placement.roi_segmentation_path if placement else None,
                    detail="Cutout evaluation has no placement ROI",
                ),
            )
        )
        self.set_details(
            (
                "Evaluation",
                {
                    "pair_id": self.group.pair_id,
                    "scope": self.group.scope,
                    "score": self.group.score,
                    "calculators": sorted(self.group.calculators),
                    "metric_differences": dict(sorted(self.group.metrics.items())),
                },
            ),
            ("RealAnomaly", real),
            ("SyntheticAnomaly", synthetic),
            ("Placement", placement),
        )
        self._render_histogram()

    def _render_histogram(self) -> None:
        self.histogram_axis.clear()
        query = self.search_query
        scope = self.scope_var.get()
        population = [
            group
            for group in self.model.evaluations
            if (scope == "all" or group.scope == scope)
            and (
                not query
                or query in group.real_anomaly_id.lower()
                or query in group.synthetic_anomaly_id.lower()
                or query in (group.placement_id or "").lower()
            )
        ]
        active = [
            name for name, variable in self._metric_vars.items() if variable.get()
        ]
        available = sorted(
            {name for group in population for name in group.metrics}
        )
        active_available = [name for name in active if name in available]
        metric = (
            active_available[0]
            if active_available
            else (available[0] if not active and available else None)
        )
        if metric is None:
            self.histogram_axis.text(
                0.5,
                0.5,
                "No evaluation metrics available",
                ha="center",
                va="center",
                transform=self.histogram_axis.transAxes,
            )
            self.histogram_axis.set_axis_off()
        else:
            self.histogram_axis.set_axis_on()
            values = [
                group.metrics[metric]
                for group in population
                if metric in group.metrics
            ]
            self.histogram_axis.hist(values, bins=32, edgecolor="black", alpha=0.7)
            if self.group and metric in self.group.metrics:
                self.histogram_axis.axvline(
                    self.group.metrics[metric], color="#ef476f", linewidth=2
                )
            self.histogram_axis.set_title(f"Difference distribution: {metric}")
            self.histogram_axis.set_xlabel("Absolute difference")
            self.histogram_axis.set_ylabel("Pairs")
        self.histogram_canvas.draw_idle()

    def previous(self) -> None:
        self._move(-1)

    def next(self) -> None:
        self._move(1)

    def _move(self, delta: int) -> None:
        if not self._items:
            return
        selected = self.tree.selection()
        try:
            index = self._items.index(selected[0])
        except (IndexError, ValueError):
            index = 0
        target = self._items[min(max(index + delta, 0), len(self._items) - 1)]
        self.tree.selection_set(target)
        self.tree.focus(target)
        self.tree.see(target)
        self._on_select()

    def close(self) -> None:
        super().close()
        plt.close(self.histogram_figure)


def _short(value: str) -> str:
    return value if len(value) <= 14 else value[-12:]
