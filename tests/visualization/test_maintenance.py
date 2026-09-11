"""Tests for consistent record and artifact maintenance."""

from hybrid_sample_generator.visualization.maintenance import StudyMaintenance
from tests.visualization.support import VisualizerStudyTestCase


class VisualizerMaintenanceTests(VisualizerStudyTestCase):
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
