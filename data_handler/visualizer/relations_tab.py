from __future__ import annotations

from tkinter import messagebox, ttk

from data_handler.visualizer.common import EntityBrowserTab, insert_tree_scrollbars
from data_handler.visualizer.maintenance import StudyMaintenance
from data_handler.visualizer.queries import StudyBrowserModel
from data_handler.visualizer.rendering import ArrayCache, Marker, PanelSpec
from data_handler.visualizer.state import SelectionController


class RelationsTab(EntityBrowserTab):
    def __init__(
        self,
        master,
        model: StudyBrowserModel,
        cache: ArrayCache,
        selection: SelectionController,
        maintenance: StudyMaintenance,
        *,
        on_data_changed,
    ) -> None:
        self.model = model
        self.selection = selection
        self.maintenance = maintenance
        self.on_data_changed = on_data_changed
        self._item_payload = {}
        self._loaded = set()
        self._selected_entity = None
        super().__init__(master, cache=cache, rows=1, columns=2)

        ttk.Label(
            self.side,
            text="Repository relationships",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 6))
        self.tree = insert_tree_scrollbars(self.side, height=22)
        self.tree.bind("<<TreeviewOpen>>", self._on_open)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.build_display_controls()

        self.delete_button = ttk.Button(
            self.details_frame,
            text="Archive and remove selected entity…",
            command=self._delete_selected,
            state="disabled",
        )
        self.delete_button.pack(fill="x", pady=(8, 0))
        self.refresh()

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._item_payload.clear()
        self._loaded.clear()
        self._selected_entity = None
        self.delete_button.configure(state="disabled")
        query = self.search_query
        first = None
        for original in self.model.originals:
            if query and query not in self._original_haystack(original.id):
                continue
            item = self._insert_entity(
                "",
                "original",
                original.id,
                f"Original: {original.source_name}",
                has_children=bool(
                    self.model.reals_by_original.get(original.id)
                    or self.model.hybrids_by_original.get(original.id)
                ),
            )
            if first is None:
                first = item
        if first:
            self.tree.selection_set(first)
            self.tree.focus(first)
            self.tree.see(first)
            self._on_select()
        else:
            self.image_grid.set_specs(())
            self.set_details(("Selection", None))

    def _insert_entity(
        self,
        parent,
        kind: str,
        entity_id,
        label: str,
        *,
        has_children: bool,
    ):
        item = self.tree.insert(parent, "end", text=label, open=False)
        self._item_payload[item] = (kind, entity_id)
        if has_children:
            self.tree.insert(item, "end", text="Loading…")
        return item

    def _on_open(self, _event=None) -> None:
        item = self.tree.focus()
        if not item or item in self._loaded or item not in self._item_payload:
            return
        for child in self.tree.get_children(item):
            self.tree.delete(child)
        kind, entity_id = self._item_payload[item]
        self._load_children(item, kind, entity_id)
        self._loaded.add(item)

    def _load_children(self, parent, kind: str, entity_id) -> None:
        if kind == "original":
            reals = self.model.reals_by_original.get(entity_id, ())
            hybrids = self.model.hybrids_by_original.get(entity_id, ())
            candidates = self.model.match_candidates(entity_id)
            if reals:
                self._insert_entity(
                    parent,
                    "real_group",
                    entity_id,
                    f"Real anomalies ({len(reals)})",
                    has_children=True,
                )
            if hybrids:
                self._insert_entity(
                    parent,
                    "hybrid_group",
                    entity_id,
                    f"Hybrid samples ({len(hybrids)})",
                    has_children=True,
                )
            if candidates:
                self._insert_entity(
                    parent,
                    "match_group",
                    entity_id,
                    f"Matching cache ({len(candidates)})",
                    has_children=True,
                )
        elif kind == "real_group":
            for real in self.model.reals_by_original.get(entity_id, ()):
                variants = self.model.synthetics_by_real.get(real.id, ())
                self._insert_entity(
                    parent,
                    "real",
                    real.id,
                    f"Real component {real.component_index}: {real.id}",
                    has_children=bool(variants),
                )
        elif kind == "hybrid_group":
            for hybrid in self.model.hybrids_by_original.get(entity_id, ()):
                placements = self.model.placements_by_hybrid.get(hybrid.id, ())
                self._insert_entity(
                    parent,
                    "hybrid",
                    hybrid.id,
                    f"Hybrid {hybrid.variant_index}: {hybrid.status}",
                    has_children=bool(placements),
                )
        elif kind == "match_group":
            for index, candidate in enumerate(self.model.match_candidates(entity_id)):
                score = "–" if candidate.score is None else f"{candidate.score:.4g}"
                label = (
                    f"{'valid' if candidate.is_valid else 'rejected'} · "
                    f"score {score} · {_short(candidate.real_anomaly_id)}"
                )
                key = (
                    candidate.original_sample_id,
                    candidate.real_anomaly_id,
                    candidate.matcher_signature,
                    index,
                )
                self._insert_entity(
                    parent,
                    "match_candidate",
                    key,
                    label,
                    has_children=False,
                )
        elif kind == "real":
            for synthetic in self.model.synthetics_by_real.get(entity_id, ()):
                self._insert_entity(
                    parent,
                    "synthetic",
                    synthetic.id,
                    f"Synthetic variant {synthetic.variant_index}: {synthetic.id}",
                    has_children=False,
                )
        elif kind == "hybrid":
            for placement in self.model.placements_by_hybrid.get(entity_id, ()):
                self._insert_entity(
                    parent,
                    "placement",
                    placement.id,
                    f"Placement {placement.order_index}: {placement.id}",
                    has_children=True,
                )
        elif kind == "placement":
            placement = self.model.placement_by_id[entity_id]
            synthetic = self.model.synthetic_by_id[placement.synthetic_anomaly_id]
            real = self.model.real_by_id[synthetic.real_anomaly_id]
            self._insert_entity(
                parent,
                "synthetic",
                synthetic.id,
                f"Synthetic reference: {synthetic.id}",
                has_children=False,
            )
            self._insert_entity(
                parent,
                "real",
                real.id,
                f"Real reference: {real.id}",
                has_children=bool(self.model.synthetics_by_real.get(real.id)),
            )

    def _on_select(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected or selected[0] not in self._item_payload:
            return
        kind, entity_id = self._item_payload[selected[0]]
        if kind.endswith("_group"):
            self._selected_entity = None
            self.delete_button.configure(state="disabled")
            self.image_grid.set_specs(())
            self.set_details(
                (
                    "Relationship group",
                    {"kind": kind, "original_sample_id": entity_id},
                )
            )
            return
        record, specs = self._record_and_specs(kind, entity_id)
        self._selected_entity = kind, entity_id
        self.delete_button.configure(
            state="disabled" if kind == "match_candidate" else "normal"
        )
        self.image_grid.set_specs(specs)
        self.set_details((type(record).__name__, record))
        self._publish_selection(kind, entity_id)

    def _record_and_specs(self, kind: str, entity_id):
        if kind == "original":
            record = self.model.original_by_id[entity_id]
            specs = (
                PanelSpec("Original image", record.image_path, record.segmentation_path),
            )
        elif kind == "real":
            record = self.model.real_by_id[entity_id]
            specs = (
                PanelSpec("Real anomaly", record.image_path, record.segmentation_path),
                PanelSpec("Real source ROI", record.roi_image_path, record.roi_segmentation_path),
            )
        elif kind == "synthetic":
            record = self.model.synthetic_by_id[entity_id]
            specs = (
                PanelSpec("Synthetic anomaly", record.image_path, record.segmentation_path),
            )
        elif kind == "hybrid":
            record = self.model.hybrid_by_id[entity_id]
            original = self.model.original_by_id[record.original_sample_id]
            specs = (
                PanelSpec("Original control", original.image_path, original.segmentation_path),
                PanelSpec(
                    "Hybrid sample",
                    record.image_path,
                    record.segmentation_path,
                    reference_path=original.image_path,
                    detail=f"Status: {record.status}",
                ),
            )
        elif kind == "match_candidate":
            original_id, real_id, _signature, index = entity_id
            record = self.model.match_candidates(original_id)[index]
            real = self.model.real_by_id[real_id]
            original = self.model.original_by_id[original_id]
            markers = (
                (Marker(record.position, "match", "#ef476f", True),)
                if record.position is not None
                else ()
            )
            specs = (
                PanelSpec(
                    "Matched original",
                    original.image_path,
                    original.segmentation_path,
                    markers=markers,
                ),
                PanelSpec(
                    "Candidate real ROI",
                    real.roi_image_path,
                    real.roi_segmentation_path,
                ),
            )
        else:
            record = self.model.placement_by_id[entity_id]
            synthetic = self.model.synthetic_by_id[record.synthetic_anomaly_id]
            specs = (
                PanelSpec(
                    "Fused placement ROI",
                    record.roi_image_path,
                    record.roi_segmentation_path,
                ),
                PanelSpec("Synthetic anomaly", synthetic.image_path, synthetic.segmentation_path),
            )
        return record, specs

    def _publish_selection(self, kind: str, entity_id) -> None:
        values = {
            "original_sample_id": None,
            "real_anomaly_id": None,
            "synthetic_anomaly_id": None,
            "hybrid_sample_id": None,
            "placement_id": None,
        }
        if kind == "original":
            values["original_sample_id"] = entity_id
        elif kind == "real":
            real = self.model.real_by_id[entity_id]
            values.update(
                original_sample_id=real.original_sample_id,
                real_anomaly_id=real.id,
            )
        elif kind == "synthetic":
            synthetic = self.model.synthetic_by_id[entity_id]
            real = self.model.real_by_id[synthetic.real_anomaly_id]
            values.update(
                original_sample_id=real.original_sample_id,
                real_anomaly_id=real.id,
                synthetic_anomaly_id=synthetic.id,
            )
        elif kind == "hybrid":
            hybrid = self.model.hybrid_by_id[entity_id]
            values.update(
                original_sample_id=hybrid.original_sample_id,
                hybrid_sample_id=hybrid.id,
            )
        elif kind == "match_candidate":
            original_id, real_id, _signature, _index = entity_id
            values.update(
                original_sample_id=original_id,
                real_anomaly_id=real_id,
            )
        else:
            context = self.model.placement_context(entity_id)
            hybrid = self.model.hybrid_by_id[context.placement.hybrid_sample_id]
            values.update(
                original_sample_id=hybrid.original_sample_id,
                real_anomaly_id=context.real.id,
                synthetic_anomaly_id=context.synthetic.id,
                hybrid_sample_id=hybrid.id,
                placement_id=context.placement.id,
            )
        self.selection.update(source="relations", **values)

    def _delete_selected(self) -> None:
        if self._selected_entity is None:
            return
        kind, entity_id = self._selected_entity
        try:
            impact = self.maintenance.preview_removal(kind, entity_id)
        except Exception as exc:
            messagebox.showerror("Cannot inspect dependencies", str(exc), parent=self)
            return
        confirmed = messagebox.askyesno(
            "Archive and remove",
            f"Remove {kind} {entity_id}?\n\n{impact.describe()}\n\n"
            "Artifact files will be moved into the study's .trash folder.",
            parent=self,
        )
        if not confirmed:
            return
        try:
            trash_path = self.maintenance.archive_and_remove(impact)
            self.cache.clear()
            self.on_data_changed()
        except Exception as exc:
            messagebox.showerror("Removal failed", str(exc), parent=self)
            return
        messagebox.showinfo(
            "Removal complete",
            f"Removed records: {impact.record_count}\nArchived artifacts: {trash_path}",
            parent=self,
        )

    def _original_haystack(self, original_id: str) -> str:
        original = self.model.original_by_id[original_id]
        values = [original.source_name, original.id]
        for real in self.model.reals_by_original.get(original_id, ()):
            values.append(real.id)
            values.extend(
                synthetic.id
                for synthetic in self.model.synthetics_by_real.get(real.id, ())
            )
        for hybrid in self.model.hybrids_by_original.get(original_id, ()):
            values.append(hybrid.id)
            for placement in self.model.placements_by_hybrid.get(hybrid.id, ()):
                values.extend((placement.id, placement.synthetic_anomaly_id))
        return " ".join(values).lower()

    def previous(self) -> None:
        self._move(-1)

    def next(self) -> None:
        self._move(1)

    def _move(self, delta: int) -> None:
        items = [
            item
            for item in self._walk_tree("")
            if item in self._item_payload
            and not self._item_payload[item][0].endswith("_group")
        ]
        if not items:
            return
        selected = self.tree.selection()
        try:
            index = items.index(selected[0])
        except (IndexError, ValueError):
            index = 0
        target = items[min(max(index + delta, 0), len(items) - 1)]
        self.tree.selection_set(target)
        self.tree.focus(target)
        self.tree.see(target)
        self._on_select()

    def _walk_tree(self, parent):
        for item in self.tree.get_children(parent):
            yield item
            if item in self._loaded:
                yield from self._walk_tree(item)


def _short(value: str) -> str:
    return value if len(value) <= 18 else value[-16:]
