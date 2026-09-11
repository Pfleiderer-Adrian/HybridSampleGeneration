"""Tests for evaluation result visualization."""

from dataclasses import replace
from unittest.mock import Mock

from matplotlib.figure import Figure

from hybrid_sample_generator.visualization.queries import filter_evaluation_groups
from hybrid_sample_generator.visualization.rendering import ArrayCache, render_panel
from hybrid_sample_generator.visualization.state import SelectionController
from hybrid_sample_generator.visualization.tabs.evaluation import EvaluationTab
from tests.visualization.support import VisualizerStudyTestCase


class VisualizerEvaluationTests(VisualizerStudyTestCase):
    def _evaluation_tab(self, group):
        tab = EvaluationTab.__new__(EvaluationTab)
        tab.model = self.model
        tab.selection = SelectionController()
        tab.group = None
        tab.preview_placement = None
        tab._placement_options = ()
        tab.tree = Mock()
        tab.tree.selection.return_value = ("selected",)
        tab._item_group = {"selected": group}
        tab.slice_var = Mock()
        tab.image_grid = Mock()
        tab.placement_choice = Mock()
        tab.set_details = Mock()
        tab._render_histogram = Mock()
        tab._on_select()
        return tab

    def test_cutout_evaluation_previews_linked_roi_and_can_switch_placements(self):
        group = next(g for g in self.model.evaluations if g.synthetic_anomaly_id == "synthetic-0" and not g.placement_id)
        planned = self.model.placement_by_id["placement-0"]
        self.store.resolve(planned.roi_image_path).unlink()
        tab = self._evaluation_tab(group)
        self.assertEqual(tab.preview_placement.id, "placement-1")
        self.assertEqual(len(tab._placement_options), 2)
        spec = tab.image_grid.set_specs.call_args.args[0][3]
        self.assertEqual(spec.image_path, self.model.placement_by_id["placement-1"].roi_image_path)
        axis = Figure().subplots()
        _, status = render_panel(
            axis, spec, ArrayCache(self.store), slice_index=0, contrast=1,
            channel="auto", show_mask=False, mask_opacity=0.45,
        )
        self.assertNotIn("missing", status)
        self.assertNotIn("error", status)
        self.assertEqual(tab.selection.state.placement_id, "placement-1")
        self.assertEqual(tab.selection.state.hybrid_sample_id, "hybrid-generated")
        self.assertEqual(tab.selection.state.original_sample_id, "original-control")

        metrics = dict(group.metrics)
        tab.placement_choice.current.return_value = 1
        tab._placement_changed()
        self.assertEqual(tab.preview_placement.id, "placement-0")
        self.assertEqual(tab.selection.state.placement_id, "placement-0")
        self.assertEqual(group.scope, "cutout")
        self.assertEqual(group.metrics, metrics)
        self.assertIsNone(group.placement_id)
        spec = tab.image_grid.set_specs.call_args.args[0][3]
        self.assertIn("artifact is missing", spec.detail)
        tab._on_select()
        self.assertEqual(tab.preview_placement.id, "placement-0")

    def test_placement_evaluation_stays_bound_to_its_evaluated_placement(self):
        group = next(g for g in self.model.evaluations if g.placement_id == "placement-1")
        evaluated = self.model.placement_by_id[group.placement_id]
        self.store.resolve(evaluated.roi_image_path).unlink()
        tab = self._evaluation_tab(group)
        self.assertEqual(tab._placement_options, (evaluated,))
        self.assertEqual(tab.preview_placement.id, group.placement_id)
        tab.placement_choice.configure.assert_called_with(
            values=(f"{evaluated.id} · {evaluated.hybrid_sample_id}",), state="disabled",
        )
        self.model.placement_by_id.pop(group.placement_id)
        tab._on_select()
        self.assertIsNone(tab.preview_placement)
        self.assertEqual(tab._placement_options, ())
        spec = tab.image_grid.set_specs.call_args.args[0][3]
        self.assertIsNone(spec.image_path)
        self.assertIn("no longer available", spec.detail)

    def test_cutout_roi_placeholder_distinguishes_unmaterialized_and_unplaced(self):
        group = next(g for g in self.model.evaluations if g.synthetic_anomaly_id == "synthetic-1")
        placement = self.model.placement_by_id["placement-2"]
        self.model.placements_by_synthetic[group.synthetic_anomaly_id] = [
            replace(placement, roi_image_path=None, roi_segmentation_path=None)
        ]
        tab = self._evaluation_tab(group)
        spec = tab.image_grid.set_specs.call_args.args[0][3]
        self.assertIsNone(spec.image_path)
        self.assertIn("no saved fused ROI yet", spec.detail)
        self.model.placements_by_synthetic[group.synthetic_anomaly_id] = []
        tab._on_select()
        self.assertIsNone(tab.preview_placement)
        self.assertIsNone(tab.selection.state.placement_id)
        spec = tab.image_grid.set_specs.call_args.args[0][3]
        self.assertIn("No placement registered", spec.detail)

    def test_evaluation_rows_are_grouped_and_filterable(self):
        cutout = next(
            group
            for group in self.model.evaluations
            if group.synthetic_anomaly_id == "synthetic-0" and not group.placement_id
        )
        self.assertEqual(set(cutout.metrics), {"Contrast", "Volume"})
        self.assertEqual(len(cutout.calculators), 2)

        filtered = filter_evaluation_groups(
            self.model.evaluations,
            metrics=("Contrast",),
            top_percent=34,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].placement_id, "placement-1")
        self.assertEqual([group.score for group in self.model.evaluations], [0.0] * 3)
