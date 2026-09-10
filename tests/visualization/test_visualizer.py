import csv
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np

from hybrid_sample_generator.visualization.tabs.anomalies import AnomaliesTab
from hybrid_sample_generator.visualization.widgets import ImageGrid
from hybrid_sample_generator.visualization.tabs.datasource import DatasourceTab
from hybrid_sample_generator.visualization.tabs.evaluation import EvaluationTab
from hybrid_sample_generator.visualization.maintenance import StudyMaintenance
from hybrid_sample_generator.visualization.queries import (
    StudyBrowserModel,
    filter_evaluation_groups,
)
from hybrid_sample_generator.visualization.rendering import (
    ArrayCache,
    PanelSpec,
    render_panel,
    display_mask_plane,
    display_plane,
    normalize_for_display,
)
from hybrid_sample_generator.visualization.state import SelectionController
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.domain.records import (
    HybridSample,
    MatchCandidate,
    OriginalSample,
    Placement,
    RealAnomaly,
    SyntheticAnomaly,
)
from hybrid_sample_generator.persistence.study_repository import StudyRepository


class VisualizerRenderingTests(unittest.TestCase):
    def test_rgb_and_channel_rendering_preserve_expected_dimensions(self):
        rgb = np.zeros((3, 5, 7), dtype=np.float32)
        rgb[0] = 255
        rgb[1] = 128

        automatic = display_plane(rgb)
        self.assertEqual(automatic.image.shape, (5, 7, 3))
        self.assertTrue(np.all(automatic.image[..., 0] == 255))
        self.assertTrue(np.all(automatic.image[..., 1] == 128))

        green = display_plane(rgb, channel="1")
        self.assertEqual(green.image.shape, (5, 7))
        self.assertTrue(np.all(green.image == 128))

        normalized = normalize_for_display(automatic.image)
        self.assertEqual(normalized.shape, (5, 7, 3))
        self.assertGreater(float(normalized[..., 0].mean()), 0.99)
        self.assertGreater(float(normalized[..., 1].mean()), 0.45)
        self.assertLess(float(normalized[..., 2].mean()), 0.01)

    def test_volume_and_mask_rendering_share_slice_coordinates(self):
        volume = np.zeros((3, 4, 5, 7), dtype=np.float32)
        volume[0, 2] = 4
        mask = np.zeros((1, 4, 5, 7), dtype=np.uint8)
        mask[:, 2, 1:3, 2:5] = 1

        image_plane = display_plane(volume, slice_index=2)
        mask_plane = display_mask_plane(mask, slice_index=2)

        self.assertEqual(image_plane.image.shape, (5, 7, 3))
        self.assertEqual(image_plane.depth, 4)
        self.assertEqual(image_plane.slice_index, 2)
        self.assertEqual(mask_plane.image.shape, (5, 7))
        self.assertEqual(int(mask_plane.image.sum()), 6)

    def test_cutout_window_excludes_padding_and_preserves_shared_scale(self):
        reference = np.full((100, 100), -1000.0, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:50, 40:50] = 2
        reference[mask > 0] = np.linspace(1, 2, 100)
        display = normalize_for_display(reference, reference_mask=mask)
        foreground = display[mask > 0]
        self.assertLess(float(foreground.min()), 0.01)
        self.assertGreater(float(foreground.max()), 0.99)
        self.assertTrue(np.all(display[mask == 0] == 0))

        synthetic = np.full((7, 9), 1.5, dtype=np.float32)
        shared = normalize_for_display(
            synthetic, reference=reference, reference_mask=mask
        )
        np.testing.assert_allclose(shared, 0.5, atol=1e-6)

    def test_masked_window_handles_rgb_constant_and_missing_foreground(self):
        rgb = np.zeros((8, 10, 3), dtype=np.float32)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:4, 3:5] = 1
        rgb[mask > 0] = [10, 20, 30]
        display = normalize_for_display(rgb, reference_mask=mask)
        np.testing.assert_allclose(display[mask > 0], [[0, 0.5, 1]] * 4)

        flat = np.zeros((8, 10), dtype=np.float32)
        flat[mask > 0] = 5
        display = normalize_for_display(flat, reference_mask=mask)
        self.assertTrue(np.all(display[mask > 0] == 1))
        for fallback_mask in (np.zeros_like(mask), np.ones((2, 3)), np.full(mask.shape, np.nan)):
            with self.subTest(mask_shape=fallback_mask.shape):
                np.testing.assert_array_equal(
                    normalize_for_display(flat, reference_mask=fallback_mask),
                    normalize_for_display(flat),
                )
        invalid = np.full_like(flat, np.nan)
        self.assertTrue(np.all(normalize_for_display(invalid, reference_mask=mask) == 0))

    def test_volume_render_uses_reference_mask_with_overlay_disabled(self):
        reference = np.full((1, 3, 10, 10), -1000.0, dtype=np.float32)
        mask = np.zeros_like(reference)
        mask[0, 1, 3:5, 3:5] = 1
        reference[0, 1, 3:5, 3:5] = [[1, 2], [3, 4]]
        synthetic = np.full((1, 3, 6, 7), 2.5, dtype=np.float32)
        cache = Mock()
        cache.get.side_effect = lambda path: {"reference-mask": mask}.get(path)
        spec = PanelSpec(
            "Synthetic anomaly", image=synthetic, reference=reference,
            reference_mask_path="reference-mask",
        )
        axis = Figure().subplots()
        depth, status = render_panel(
            axis, spec, cache, slice_index=1, contrast=1, channel="auto",
            show_mask=False, mask_opacity=0.45,
        )
        self.assertEqual(depth, 3)
        self.assertNotIn("error", status)
        self.assertEqual(len(axis.images), 1)
        np.testing.assert_allclose(axis.images[0].get_array(), 0.5, atol=1e-6)


class VisualizerNavigationTests(unittest.TestCase):
    def setUp(self):
        self.grid = ImageGrid.__new__(ImageGrid)
        grid = self.grid
        grid.figure = Figure()
        grid.canvas = FigureCanvasAgg(grid.figure)
        grid.axes = tuple(grid.figure.subplots(1, 2))
        grid.cache = Mock()
        grid.cache.get.return_value = None
        grid.slice_index = 0
        grid.contrast = 1.0
        grid.channel = "auto"
        grid.show_mask = False
        grid.mask_opacity = 0.45
        grid.on_depth_changed = None
        grid._views = {}
        grid._full_views = {}
        grid._pan_axis = None
        grid._scroll_callback = Mock()
        image = np.arange(200, dtype=np.float32).reshape(1, 2, 10, 10)
        grid.set_specs((PanelSpec("First", image=image), PanelSpec("Second", image=image)))
        grid.canvas.draw()

    def _scroll(self, **kwargs):
        event = SimpleNamespace(
            inaxes=self.grid.axes[0], step=1, key=None, xdata=3.0, ydata=4.0,
        )
        event.__dict__.update(kwargs)
        self.grid._on_scroll(event)

    def test_zoom_is_cursor_centered_and_survives_display_changes(self):
        grid = self.grid
        axis, other = grid.axes
        before_x, before_y = axis.get_xlim(), axis.get_ylim()
        other_limits = other.get_xlim(), other.get_ylim()
        self._scroll()
        expected_x = [3 + (x - 3) / 1.2 for x in before_x]
        expected_y = [4 + (y - 4) / 1.2 for y in before_y]
        np.testing.assert_allclose(axis.get_xlim(), expected_x)
        np.testing.assert_allclose(axis.get_ylim(), expected_y)
        self.assertEqual((other.get_xlim(), other.get_ylim()), other_limits)
        self.assertGreater(axis.get_ylim()[0], axis.get_ylim()[1])
        grid.slice_index = 1
        grid.contrast = 2
        grid.render()
        np.testing.assert_allclose(axis.get_xlim(), expected_x)
        np.testing.assert_allclose(axis.get_ylim(), expected_y)
        grid.reset_view()
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), (before_x, before_y))

    def test_shift_wheel_changes_slice_and_background_scroll_is_ignored(self):
        axis = self.grid.axes[0]
        limits = axis.get_xlim(), axis.get_ylim()
        self._scroll(key="shift", step=-1)
        self.grid._scroll_callback.assert_called_once_with(-1)
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), limits)
        self._scroll(inaxes=None)
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), limits)
        self._scroll()
        self.grid.set_specs(self.grid.specs)
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), limits)

    def test_drag_pans_and_double_click_resets_selected_panel(self):
        grid = self.grid
        axis = grid.axes[0]
        self._scroll()
        before = axis.get_xlim()
        x, y = axis.transData.transform((4, 4))
        grid._on_press(SimpleNamespace(inaxes=axis, button=1, dblclick=False, x=x, y=y))
        grid._on_motion(SimpleNamespace(x=x + 20, y=y + 10))
        grid._on_release(None)
        self.assertNotEqual(axis.get_xlim(), before)
        self.assertAlmostEqual(axis.get_xlim()[1] - axis.get_xlim()[0], before[1] - before[0])
        panned = axis.get_xlim(), axis.get_ylim()
        grid.render()
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), panned)
        grid._on_press(SimpleNamespace(inaxes=axis, button=1, dblclick=True))
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), grid._full_views[axis])


class VisualizerModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = StudyRepository(self.root / "artifacts.sqlite")
        self.store = ArtifactStore(self.root)
        self._create_study()
        self.csv_path = self.root / "evaluation_results" / "metric_diffs.csv"
        self._create_evaluation_csv()
        self.model = StudyBrowserModel(
            self.repository,
            self.store,
            metric_csv_path=str(self.csv_path),
        )

    def tearDown(self):
        self.temporary.cleanup()

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

    def test_removal_preview_cascades_to_complete_hybrids_and_archives_files(self):
        maintenance = StudyMaintenance(self.model)
        impact = maintenance.preview_removal("synthetic", "synthetic-0")

        self.assertEqual(impact.synthetic_ids, ("synthetic-0",))
        self.assertEqual(
            set(impact.hybrid_ids), {"hybrid-planned", "hybrid-generated"}
        )
        self.assertEqual(
            set(impact.placement_ids),
            {"placement-0", "placement-1", "placement-2"},
        )
        source_image = self.store.resolve(
            self.model.synthetic_by_id["synthetic-0"].image_path
        )
        self.assertTrue(source_image.is_file())

        trash = maintenance.archive_and_remove(impact)

        self.assertTrue(trash.is_dir())
        self.assertFalse(source_image.exists())
        self.assertNotIn("synthetic-0", self.model.synthetic_by_id)
        self.assertIn("synthetic-1", self.model.synthetic_by_id)
        self.assertEqual(len(self.model.hybrids), 0)
        self.assertEqual(len(self.model.placements), 0)

    def _save(self, entity_type, entity_id, role, array):
        return self.store.save_entity_array(entity_type, entity_id, role, array)

    def _create_study(self):
        image = np.zeros((3, 8, 10), dtype=np.float32)
        image[0] = 255
        mask = np.zeros((1, 8, 10), dtype=np.uint8)
        mask[:, 2:5, 3:6] = 1
        empty_mask = np.zeros_like(mask)
        original_anomaly = OriginalSample(
            "original-anomaly",
            "anomaly.png",
            self._save("original_samples", "original-anomaly", "image", image),
            self._save(
                "original_samples", "original-anomaly", "segmentation", mask
            ),
            2,
            True,
            True,
            0,
        )
        original_control = OriginalSample(
            "original-control",
            "control.png",
            self._save("original_samples", "original-control", "image", image / 2),
            self._save(
                "original_samples", "original-control", "segmentation", empty_mask
            ),
            2,
            False,
            True,
            1,
        )
        self.repository.replace_original_samples([original_anomaly, original_control])

        roi = image[:, 1:5, 2:6]
        roi_mask = mask[:, 1:5, 2:6]
        real = RealAnomaly(
            "real-0",
            original_anomaly.id,
            0,
            self._save("real_anomalies", "real-0", "image", roi),
            self._save("real_anomalies", "real-0", "segmentation", roi_mask),
            self._save("real_anomalies", "real-0", "roi_image", roi),
            self._save("real_anomalies", "real-0", "roi_segmentation", roi_mask),
            2,
            None,
            0.4,
            0.5,
            {"roi_shape": [4, 4]},
        )
        self.repository.upsert_real_anomaly(real)

        synthetics = []
        for index in range(2):
            synthetic_id = f"synthetic-{index}"
            synthetics.append(
                SyntheticAnomaly(
                    synthetic_id,
                    real.id,
                    index,
                    self._save(
                        "synthetic_anomalies",
                        synthetic_id,
                        "image",
                        roi + index,
                    ),
                    self._save(
                        "synthetic_anomalies",
                        synthetic_id,
                        "segmentation",
                        roi_mask,
                    ),
                    100 + index,
                )
            )
        for synthetic in synthetics:
            self.repository.upsert_synthetic_anomaly(synthetic)

        hybrid_image = image / 2
        hybrid_image[:, 4:7, 5:8] += 10
        hybrids = [
            HybridSample("hybrid-planned", original_control.id, 0),
            HybridSample(
                "hybrid-generated",
                original_control.id,
                1,
                self._save(
                    "hybrid_samples", "hybrid-generated", "image", hybrid_image
                ),
                self._save(
                    "hybrid_samples",
                    "hybrid-generated",
                    "segmentation",
                    empty_mask,
                ),
                "generated",
            ),
        ]
        placements = []
        for index, (placement_id, hybrid_id, synthetic_id) in enumerate(
            (
                ("placement-0", "hybrid-planned", "synthetic-0"),
                ("placement-1", "hybrid-generated", "synthetic-0"),
                ("placement-2", "hybrid-generated", "synthetic-1"),
            )
        ):
            placements.append(
                Placement(
                    placement_id,
                    hybrid_id,
                    synthetic_id,
                    index if hybrid_id == "hybrid-generated" else 0,
                    2,
                    None,
                    0.5,
                    0.6,
                    score=0.9 - index * 0.1,
                    method="local",
                    roi_image_path=self._save(
                        "placements", placement_id, "roi_image", roi
                    ),
                    roi_segmentation_path=self._save(
                        "placements", placement_id, "roi_segmentation", roi_mask
                    ),
                )
            )
        self.repository.replace_hybrid_plan(hybrids, placements)
        self.repository.upsert_match_candidates(
            [
                MatchCandidate(
                    original_control.id,
                    real.id,
                    "matcher-test",
                    True,
                    0.9,
                    (0.5, 0.6),
                    (4.0, 6.0),
                    (4, 4),
                )
            ]
        )

    def _create_evaluation_csv(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            (
                "synthetic-0",
                "real-0",
                "synthetic-0",
                "",
                "get_glcm_feature_diffs",
                {"Contrast": 0.2},
            ),
            (
                "synthetic-0",
                "real-0",
                "synthetic-0",
                "",
                "get_volume_feature_diffs",
                {"Volume": 2.0},
            ),
            (
                "synthetic-1",
                "real-0",
                "synthetic-1",
                "",
                "get_glcm_feature_diffs",
                {"Contrast": 0.5},
            ),
            (
                "placement-1",
                "real-0",
                "synthetic-0",
                "placement-1",
                "get_glcm_roi_feature_diffs",
                {"Contrast": 0.8, "roi_Energy": 0.4},
            ),
        )
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "pair_id",
                    "real_anomaly_id",
                    "synthetic_anomaly_id",
                    "placement_id",
                    "feature_calculator",
                    "metric_diffs",
                )
            )
            for *values, metrics in rows:
                writer.writerow((*values, json.dumps(metrics)))


if __name__ == "__main__":
    unittest.main()
