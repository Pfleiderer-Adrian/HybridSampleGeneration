from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from data_handler.visualizer.common import EntityBrowserTab, insert_tree_scrollbars
from data_handler.visualizer.queries import HybridContext, StudyBrowserModel
from data_handler.visualizer.rendering import ArrayCache, Marker, PanelSpec
from data_handler.visualizer.state import SelectionController


MARKER_COLORS = (
    "#ffd166",
    "#06d6a0",
    "#118ab2",
    "#ef476f",
    "#a78bfa",
    "#f97316",
)


class HybridsTab(EntityBrowserTab):
    def __init__(
        self,
        master,
        model: StudyBrowserModel,
        cache: ArrayCache,
        selection: SelectionController,
    ) -> None:
        self.model = model
        self.selection = selection
        self.context: HybridContext | None = None
        self._item_payload = {}
        self._leaf_items = []
        super().__init__(master, cache=cache, rows=2, columns=3)

        ttk.Label(
            self.side,
            text="Original → Hybrid → Placement",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")
        status_row = ttk.Frame(self.side)
        status_row.pack(fill="x", pady=(6, 6))
        ttk.Label(status_row, text="Status").pack(side="left")
        self.hybrid_status_filter = ttk.Combobox(
            status_row,
            values=("all", "generated", "planned", "failed"),
            state="readonly",
            width=11,
        )
        self.hybrid_status_filter.set("all")
        self.hybrid_status_filter.pack(side="right")
        self.hybrid_status_filter.bind(
            "<<ComboboxSelected>>", lambda _event: self.refresh()
        )
        self.tree = insert_tree_scrollbars(
            self.side,
            columns=("status",),
            headings=("Status",),
            height=18,
        )
        self.tree.heading("#0", text="Original / hybrid / placement")
        self.tree.column("#0", width=220, stretch=True)
        self.tree.column("status", width=72, stretch=False)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.difference_var = None
        self.build_display_controls()
        self.difference_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.side,
            text="Show absolute difference",
            variable=self.difference_var,
            command=self._show_context,
        ).pack(anchor="w", pady=(6, 0))
        self.refresh()

    def refresh(self) -> None:
        selected_key = (
            None
            if self.context is None
            else (
                self.context.hybrid.id,
                self.context.selected_placement.placement.id
                if self.context.selected_placement
                else None,
            )
        )
        if selected_key is None:
            default_context = self.model.first_hybrid_context()
            if default_context is not None:
                selected_key = (
                    default_context.hybrid.id,
                    default_context.selected_placement.placement.id
                    if default_context.selected_placement
                    else None,
                )
        self.tree.delete(*self.tree.get_children())
        self._item_payload.clear()
        self._leaf_items.clear()
        preferred_item = None
        status_filter = self.hybrid_status_filter.get() or "all"
        query = self.search_query

        for original in self.model.originals:
            matching = []
            for hybrid in self.model.hybrids_by_original.get(original.id, ()):
                if status_filter != "all" and hybrid.status != status_filter:
                    continue
                placements = self.model.placements_by_hybrid.get(hybrid.id, ())
                related_ids = []
                for placement in placements:
                    related_ids.extend((placement.id, placement.synthetic_anomaly_id))
                haystack = " ".join(
                    (original.source_name, original.id, hybrid.id, *related_ids)
                ).lower()
                if not query or query in haystack:
                    matching.append((hybrid, placements))
            if not matching:
                continue

            original_item = self.tree.insert(
                "",
                "end",
                text=original.source_name,
                values=(f"{len(matching)} hybrids",),
                open=preferred_item is None,
            )
            self._item_payload[original_item] = ("original", original.id)
            for hybrid, placements in matching:
                hybrid_item = self.tree.insert(
                    original_item,
                    "end",
                    text=f"Hybrid {hybrid.variant_index}",
                    values=(hybrid.status,),
                    open=preferred_item is None,
                )
                self._item_payload[hybrid_item] = ("hybrid", hybrid.id)
                if not placements:
                    self._leaf_items.append(hybrid_item)
                    if selected_key == (hybrid.id, None):
                        preferred_item = hybrid_item
                for placement in placements:
                    item = self.tree.insert(
                        hybrid_item,
                        "end",
                        text=f"Placement {placement.order_index}",
                        values=(placement.method,),
                    )
                    self._item_payload[item] = ("placement", placement.id)
                    self._leaf_items.append(item)
                    if selected_key == (hybrid.id, placement.id):
                        preferred_item = item

        target = preferred_item or (self._leaf_items[0] if self._leaf_items else None)
        if target:
            self.tree.selection_set(target)
            self.tree.focus(target)
            self.tree.see(target)
            self._on_select()
        else:
            self.context = None
            self.image_grid.set_specs(())
            self.set_details(("Selection", None))

    def _on_select(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        kind, entity_id = self._item_payload[selected[0]]
        if kind == "original":
            hybrid_items = self.tree.get_children(selected[0])
            if not hybrid_items:
                return
            _kind, hybrid_id = self._item_payload[hybrid_items[0]]
            context = self.model.hybrid_context(hybrid_id)
        elif kind == "hybrid":
            context = self.model.hybrid_context(entity_id)
        else:
            placement = self.model.placement_by_id[entity_id]
            context = self.model.hybrid_context(placement.hybrid_sample_id, entity_id)
        self.context = context
        self.slice_var.set(0)
        selected_placement = context.selected_placement
        self.selection.update(
            source="hybrids",
            original_sample_id=context.original.id,
            hybrid_sample_id=context.hybrid.id,
            placement_id=(
                selected_placement.placement.id if selected_placement else None
            ),
            real_anomaly_id=(selected_placement.real.id if selected_placement else None),
            synthetic_anomaly_id=(
                selected_placement.synthetic.id if selected_placement else None
            ),
        )
        self._show_context()

    def _show_context(self) -> None:
        if self.context is None:
            return
        context = self.context
        selected = context.selected_placement
        markers = tuple(
            Marker(
                placement.placement.position,
                str(placement.placement.order_index),
                MARKER_COLORS[index % len(MARKER_COLORS)],
                selected is not None
                and placement.placement.id == selected.placement.id,
            )
            for index, placement in enumerate(context.placements)
        )
        difference = None
        if self.difference_var is not None and self.difference_var.get():
            original_array = self.cache.get(context.original.image_path)
            hybrid_array = self.cache.get(context.hybrid.image_path)
            if (
                original_array is not None
                and hybrid_array is not None
                and original_array.shape == hybrid_array.shape
            ):
                difference = np.abs(
                    np.asarray(hybrid_array, dtype=np.float32)
                    - np.asarray(original_array, dtype=np.float32)
                )

        self.image_grid.set_specs(
            (
                PanelSpec(
                    "Original control",
                    context.original.image_path,
                    context.original.segmentation_path,
                    markers=markers,
                ),
                PanelSpec(
                    f"Hybrid variant {context.hybrid.variant_index}",
                    context.hybrid.image_path,
                    context.hybrid.segmentation_path,
                    reference_path=context.original.image_path,
                    markers=markers,
                    detail=f"Status: {context.hybrid.status}",
                ),
                PanelSpec(
                    "Absolute difference",
                    image=difference,
                    detail="Difference view disabled or hybrid unavailable",
                ),
                PanelSpec(
                    "Synthetic anomaly",
                    selected.synthetic.image_path if selected else None,
                    selected.synthetic.segmentation_path if selected else None,
                    reference_path=(selected.real.roi_image_path if selected else None),
                    detail="Select a placement",
                ),
                PanelSpec(
                    "Fused placement ROI",
                    selected.placement.roi_image_path if selected else None,
                    selected.placement.roi_segmentation_path if selected else None,
                    detail="Placement is not materialized",
                ),
                PanelSpec(
                    "Real source ROI",
                    selected.real.roi_image_path if selected else None,
                    selected.real.roi_segmentation_path if selected else None,
                    detail="Select a placement",
                ),
            )
        )
        placement_rows = [
            {
                "order": value.placement.order_index,
                "placement_id": value.placement.id,
                "synthetic_id": value.synthetic.id,
                "real_id": value.real.id,
                "position": value.placement.position,
                "score": value.placement.score,
                "method": value.placement.method,
            }
            for value in context.placements
        ]
        self.set_details(
            ("OriginalSample", context.original),
            ("HybridSample", context.hybrid),
            ("Placements", placement_rows),
            ("Selected Placement", selected.placement if selected else None),
            ("SyntheticAnomaly", selected.synthetic if selected else None),
            ("RealAnomaly", selected.real if selected else None),
        )

    def previous(self) -> None:
        self._move(-1)

    def next(self) -> None:
        self._move(1)

    def _move(self, delta: int) -> None:
        if not self._leaf_items:
            return
        selected = self.tree.selection()
        try:
            index = self._leaf_items.index(selected[0])
        except (IndexError, ValueError):
            index = 0
        target = self._leaf_items[min(max(index + delta, 0), len(self._leaf_items) - 1)]
        self.tree.selection_set(target)
        self.tree.focus(target)
        self.tree.see(target)
        self._on_select()
