"""Tests for record browsing and datasource filtering."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from matplotlib.figure import Figure
import numpy as np

from hybrid_sample_generator.visualization.rendering import (
    ArrayCache,
    display_plane,
    normalize_for_display,
    render_panel,
)
from hybrid_sample_generator.visualization.state import SelectionController
from hybrid_sample_generator.visualization.tabs.anomalies import AnomaliesTab
from hybrid_sample_generator.visualization.tabs.datasource import DatasourceTab
from tests.visualization.support import VisualizerStudyTestCase


class VisualizerBrowserTests(VisualizerStudyTestCase):
    def test_model_resolves_variants_placements_and_generated_default(self):
        anomaly = self.model.anomaly_context("real-0", "synthetic-1")
        self.assertEqual(anomaly.original.id, "original-anomaly")
        self.assertEqual(len(anomaly.variants), 2)
        self.assertEqual(anomaly.synthetic.variant_index, 1)

        hybrid = self.model.first_hybrid_context()
        self.assertEqual(hybrid.hybrid.id, "hybrid-generated")
        self.assertEqual(len(hybrid.placements), 2)
        self.assertEqual(hybrid.selected_placement.placement.id, "placement-1")
        self.assertEqual(hybrid.selected_placement.real.id, "real-0")

        summary = self.model.summary()
        self.assertEqual(summary["synthetic_anomalies"], 2)
        self.assertEqual(summary["hybrids"], 2)
        self.assertEqual(summary["placements"], 3)
        self.assertEqual(summary["match_candidates"], 1)
        self.assertEqual(summary["evaluation_pairs"], 3)
        candidates = self.model.match_candidates("original-control")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].real_anomaly_id, "real-0")

    def test_anomaly_panels_render_normalized_cutouts_instead_of_roi_scale(self):
        context = self.model.anomaly_context("real-0", "synthetic-0")
        mask = self.store.load_array(context.real.segmentation_path)
        real = np.full((3, 4, 4), -0.5, dtype=np.float32)
        real[:, mask[0] > 0] = np.linspace(1, 3, 27).reshape(3, 9)
        roi = np.linspace(100, 200, 48, dtype=np.float32).reshape(3, 4, 4)
        for path, array in (
            (context.real.image_path, real),
            (context.synthetic.image_path, real.copy()),
            (context.real.roi_image_path, roi),
        ):
            np.save(self.store.resolve(path), array)

        tab = SimpleNamespace(
            context=context, image_grid=Mock(), set_details=Mock(),
        )
        AnomaliesTab._show_context(tab)
        specs = tab.image_grid.set_specs.call_args.args[0]
        cache = ArrayCache(self.store)
        rendered = []
        for spec in (specs[1], specs[3]):
            self.assertEqual(spec.reference_path, context.real.image_path)
            self.assertEqual(spec.reference_mask_path, context.real.segmentation_path)
            axis = Figure().subplots()
            _, status = render_panel(
                axis, spec, cache, slice_index=0, contrast=1, channel="auto",
                show_mask=False, mask_opacity=0.45,
            )
            self.assertNotIn("error", status)
            display = np.asarray(axis.images[0].get_array())
            self.assertGreater(float(display[mask[0] > 0].mean()), 0.4)
            self.assertTrue(np.all(display[mask[0] == 0] == 0))
            rendered.append(display)
        np.testing.assert_array_equal(*rendered)
        old_display = normalize_for_display(
            display_plane(real).image, reference=display_plane(roi).image
        )
        self.assertTrue(np.all(old_display == 0))

    def test_datasource_filters_originals_without_requiring_anomalies(self):
        tab = DatasourceTab.__new__(DatasourceTab)
        tab.model = self.model
        tab.search_query = ""
        tab.kind_var = Mock()
        tab.kind_var.get.return_value = "All samples"
        tab.annotation_var = Mock()
        tab.annotation_var.get.return_value = "All annotations"
        self.assertEqual(len(tab._filtered_records()), 2)
        tab.kind_var.get.return_value = "Controls"
        self.assertEqual([r.id for r in tab._filtered_records()], ["original-control"])
        tab.search_query = "anomaly.png"
        self.assertEqual(tab._filtered_records(), [])
        tab.kind_var.get.return_value = "Anomalous"
        self.assertEqual([r.id for r in tab._filtered_records()], ["original-anomaly"])
        tab.search_query = "original-control"
        tab.kind_var.get.return_value = "All samples"
        self.assertEqual([r.id for r in tab._filtered_records()], ["original-control"])
        tab.search_query = ""
        tab.model.originals = [replace(r, is_annotated=r.has_anomaly) for r in tab.model.originals]
        tab.annotation_var.get.return_value = "Unannotated"
        self.assertEqual([r.id for r in tab._filtered_records()], ["original-control"])

    def test_datasource_selection_and_empty_filter_clear_stale_entities(self):
        tab = DatasourceTab.__new__(DatasourceTab)
        tab.model = self.model
        tab.original = None
        tab.tree = Mock()
        tab.tree.selection.return_value = ("original-control",)
        tab.tree.get_children.return_value = ("original-control",)
        tab.image_grid = Mock()
        tab.slice_var = Mock()
        tab.set_details = Mock()
        tab.count_var = Mock()
        tab.selection = SelectionController()
        tab.selection.update(real_anomaly_id="real-0", synthetic_anomaly_id="synthetic-0")
        tab._on_select()
        self.assertEqual(tab.selection.state.original_sample_id, "original-control")
        self.assertIsNone(tab.selection.state.real_anomaly_id)
        self.assertIsNone(tab.selection.state.synthetic_anomaly_id)
        self.assertEqual(tab.image_grid.slice_index, 0)
        spec = tab.image_grid.set_specs.call_args.args[0][0]
        self.assertEqual(spec.image_path, tab.original.image_path)
        self.assertEqual(spec.mask_path, tab.original.segmentation_path)
        tab._filtered_records = Mock(return_value=[])
        tab.refresh()
        self.assertIsNone(tab.original)
        self.assertIsNone(tab.selection.state.original_sample_id)
        self.assertIn("No original samples", tab.image_grid.set_specs.call_args.args[0][0].detail)
    def test_selection_controller_publishes_normalized_ids(self):
        controller = SelectionController()
        seen = []
        controller.subscribe(seen.append)
        state = controller.update(
            source="test",
            original_sample_id="original-control",
            hybrid_sample_id="hybrid-generated",
        )
        self.assertEqual(state.source, "test")
        self.assertEqual(seen[-1], state)
