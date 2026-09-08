from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from data_handler.visualizer.common import EntityBrowserTab, insert_tree_scrollbars
from data_handler.visualizer.queries import StudyBrowserModel
from data_handler.visualizer.rendering import ArrayCache, PanelSpec
from data_handler.visualizer.state import SelectionController


class DatasourceTab(EntityBrowserTab):
    """Browse all ingested originals, including sources without extracted anomalies."""

    def __init__(
        self,
        master,
        model: StudyBrowserModel,
        cache: ArrayCache,
        selection: SelectionController,
    ) -> None:
        self.model = model
        self.selection = selection
        self.original = None
        super().__init__(master, cache=cache, rows=1, columns=1)

        ttk.Label(self.side, text="Original datasource", font=("Arial", 10, "bold")).pack(
            anchor="w", pady=(0, 6)
        )
        self.kind_var = tk.StringVar(value="All samples")
        self.annotation_var = tk.StringVar(value="All annotations")
        for variable, values in (
            (self.kind_var, ("All samples", "Anomalous", "Controls")),
            (self.annotation_var, ("All annotations", "Annotated", "Unannotated")),
        ):
            dropdown = ttk.Combobox(
                self.side, textvariable=variable, values=values, state="readonly"
            )
            dropdown.pack(fill="x", pady=(0, 4))
            dropdown.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        self.count_var = tk.StringVar(value="")
        ttk.Label(self.side, textvariable=self.count_var).pack(anchor="w", pady=4)
        self.tree = insert_tree_scrollbars(
            self.side, columns=("kind", "annotated"), headings=("Type", "Annotated"),
            height=16,
        )
        self.tree.heading("#0", text="Source")
        self.tree.column("#0", width=190)
        self.tree.column("kind", width=85, stretch=False)
        self.tree.column("annotated", width=80, stretch=False)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.build_display_controls()
        self.refresh()

    def _filtered_records(self):
        kind = self.kind_var.get()
        annotation = self.annotation_var.get()
        return [
            record for record in self.model.originals
            if (kind == "All samples" or record.has_anomaly == (kind == "Anomalous"))
            and (annotation == "All annotations" or record.is_annotated == (annotation == "Annotated"))
            and (not self.search_query or self.search_query in f"{record.source_name} {record.id}".lower())
        ]

    def refresh(self) -> None:
        selected_id = self.original.id if self.original else None
        records = self._filtered_records()
        self.tree.delete(*self.tree.get_children())
        for record in records:
            self.tree.insert(
                "", "end", iid=record.id, text=record.source_name,
                values=("Anomalous" if record.has_anomaly else "Control",
                        "Yes" if record.is_annotated else "No"),
            )
        self.count_var.set(f"{len(records)} / {len(self.model.originals)} originals")
        visible_ids = {record.id for record in records}
        target = selected_id if selected_id in visible_ids else (records[0].id if records else None)
        if target is None:
            self.original = None
            self.image_grid.set_specs((PanelSpec(
                "Original source", detail="No original samples match the current filters",
            ),))
            self.set_details(("Selection", None))
            self.selection.clear(source="datasource")
            return
        self.tree.selection_set(target)
        self.tree.focus(target)
        self.tree.see(target)
        self._on_select()

    def _on_select(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        original = self.model.original_by_id.get(selected[0])
        if original is None:
            return
        if self.original is None or original.id != self.original.id:
            self.slice_var.set(0)
            self.image_grid.slice_index = 0
        self.original = original
        self.selection.update(
            source="datasource", original_sample_id=original.id,
            real_anomaly_id=None, synthetic_anomaly_id=None,
            hybrid_sample_id=None, placement_id=None,
        )
        self.image_grid.set_specs((PanelSpec(
            original.source_name, original.image_path, original.segmentation_path,
        ),))
        reals = self.model.reals_by_original.get(original.id, ())
        self.set_details(
            ("OriginalSample", original),
            ("Derived samples", {
                "real_anomalies": len(reals),
                "synthetic_variants": sum(
                    len(self.model.synthetics_by_real.get(real.id, ())) for real in reals
                ),
                "hybrids": len(self.model.hybrids_by_original.get(original.id, ())),
            }),
        )

    def previous(self) -> None:
        self._move(-1)

    def next(self) -> None:
        self._move(1)

    def _move(self, delta: int) -> None:
        items = self.tree.get_children()
        if not items:
            return
        selected = self.tree.selection()
        index = items.index(selected[0]) if selected and selected[0] in items else 0
        target = items[min(max(index + delta, 0), len(items) - 1)]
        self.tree.selection_set(target)
        self.tree.focus(target)
        self.tree.see(target)
        self._on_select()
