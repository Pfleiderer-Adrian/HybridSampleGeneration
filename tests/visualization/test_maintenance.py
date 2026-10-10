"""Tests for consistent record and artifact maintenance."""

from hybrid_sample_generator.visualization.maintenance import StudyMaintenance
from tests.visualization.support import VisualizerStudyTestCase


class VisualizerMaintenanceTests(VisualizerStudyTestCase):
    def test_placement_removal_deletes_its_complete_hybrid(self):
        maintenance = StudyMaintenance(self.model)
        impact = maintenance.preview_removal("placement", "placement-1")

        self.assertEqual(impact.hybrid_ids, ("hybrid-generated",))
        self.assertEqual(
            set(impact.placement_ids),
            {"placement-1", "placement-2"},
        )
        self.assertEqual(impact.original_ids, ())
        self.assertEqual(impact.real_ids, ())
        self.assertEqual(impact.synthetic_ids, ())
        hybrid_image = self.store.resolve(
            self.model.hybrid_by_id["hybrid-generated"].image_path
        )
        placement_image = self.store.resolve(
            self.model.placement_by_id["placement-1"].roi_image_path
        )

        trash = maintenance.archive_and_remove(impact)

        self.assertTrue(trash.is_dir())
        self.assertFalse(hybrid_image.exists())
        self.assertFalse(placement_image.exists())
        self.assertNotIn("hybrid-generated", self.model.hybrid_by_id)
        self.assertNotIn("placement-1", self.model.placement_by_id)
        self.assertNotIn("placement-2", self.model.placement_by_id)
        self.assertIn("hybrid-planned", self.model.hybrid_by_id)
        self.assertIn("placement-0", self.model.placement_by_id)

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
