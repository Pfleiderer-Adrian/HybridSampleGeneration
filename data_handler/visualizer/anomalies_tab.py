from __future__ import annotations

from tkinter import ttk

from data_handler.visualizer.common import EntityBrowserTab, insert_tree_scrollbars
from data_handler.visualizer.queries import AnomalyContext, StudyBrowserModel
from data_handler.visualizer.rendering import ArrayCache, PanelSpec
from data_handler.visualizer.state import SelectionController


class AnomaliesTab(EntityBrowserTab):
    def __init__(
        self,
        master,
        model: StudyBrowserModel,
        cache: ArrayCache,
        selection: SelectionController,
    ) -> None:
        self.model = model
        self.selection = selection
        self.context: AnomalyContext | None = None
        self._item_payload = {}
        self._leaf_items = []
        super().__init__(master, cache=cache, rows=2, columns=2)

        ttk.Label(
            self.side,
            text="Original → Real anomaly → Synthetic variant",
            font=("Arial", 10, "bold"),
            wraplength=280,
        ).pack(anchor="w", pady=(0, 6))
        self.tree = insert_tree_scrollbars(self.side, height=20)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.build_display_controls()
        self.refresh()

    def refresh(self) -> None:
        selected_key = (
            None
            if self.context is None
            else (
                self.context.real.id,
                self.context.synthetic.id if self.context.synthetic else None,
            )
        )
        self.tree.delete(*self.tree.get_children())
        self._item_payload.clear()
        self._leaf_items.clear()
        preferred_item = None
        query = self.search_query

        for original in self.model.originals:
            reals = self.model.reals_by_original.get(original.id, ())
            matching_reals = []
            for real in reals:
                variants = self.model.synthetics_by_real.get(real.id, ())
                haystack = " ".join(
                    (
                        original.source_name,
                        original.id,
                        real.id,
                        *(variant.id for variant in variants),
                    )
                ).lower()
                if not query or query in haystack:
                    matching_reals.append((real, variants))
            if not matching_reals:
                continue

            original_item = self.tree.insert(
                "",
                "end",
                text=f"{original.source_name} ({len(matching_reals)} components)",
                open=preferred_item is None,
            )
            self._item_payload[original_item] = ("original", original.id)
            for real, variants in matching_reals:
                real_item = self.tree.insert(
                    original_item,
                    "end",
                    text=f"Real component {real.component_index} · {real.id}",
                    open=preferred_item is None,
                )
                self._item_payload[real_item] = ("real", real.id)
                if not variants:
                    self._leaf_items.append(real_item)
                    if selected_key == (real.id, None):
                        preferred_item = real_item
                for synthetic in variants:
                    synthetic_item = self.tree.insert(
                        real_item,
                        "end",
                        text=f"Variant {synthetic.variant_index} · {synthetic.id}",
                    )
                    self._item_payload[synthetic_item] = ("synthetic", synthetic.id)
                    self._leaf_items.append(synthetic_item)
                    if selected_key == (real.id, synthetic.id):
                        preferred_item = synthetic_item

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
            real_items = self.tree.get_children(selected[0])
            if not real_items:
                return
            _kind, real_id = self._item_payload[real_items[0]]
            context = self.model.anomaly_context(real_id)
        elif kind == "real":
            context = self.model.anomaly_context(entity_id)
        else:
            synthetic = self.model.synthetic_by_id[entity_id]
            context = self.model.anomaly_context(synthetic.real_anomaly_id, entity_id)
        self.context = context
        self.slice_var.set(0)
        self.selection.update(
            source="anomalies",
            original_sample_id=context.original.id,
            real_anomaly_id=context.real.id,
            synthetic_anomaly_id=(context.synthetic.id if context.synthetic else None),
            hybrid_sample_id=None,
            placement_id=None,
        )
        self._show_context()

    def _show_context(self) -> None:
        if self.context is None:
            return
        real = self.context.real
        synthetic = self.context.synthetic
        self.image_grid.set_specs(
            (
                PanelSpec(
                    "Original source",
                    self.context.original.image_path,
                    self.context.original.segmentation_path,
                ),
                PanelSpec(
                    "Extracted real anomaly",
                    real.image_path,
                    real.segmentation_path,
                    reference_path=real.roi_image_path,
                ),
                PanelSpec(
                    "Real source ROI",
                    real.roi_image_path,
                    real.roi_segmentation_path,
                ),
                PanelSpec(
                    "Synthetic variant",
                    synthetic.image_path if synthetic else None,
                    synthetic.segmentation_path if synthetic else None,
                    reference_path=real.roi_image_path,
                    detail="No synthetic variant registered",
                ),
            )
        )
        self.set_details(
            ("OriginalSample", self.context.original),
            ("RealAnomaly", real),
            ("SyntheticAnomaly", synthetic),
            ("Variant count", {"variants": len(self.context.variants)}),
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
