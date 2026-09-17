"""Removal cascades and rollback of archive/database failures."""
import shutil
import sqlite3
from unittest.mock import patch

from hybrid_sample_generator.visualization.maintenance import StudyMaintenance
from tests.visualization.support import VisualizerStudyTestCase


class MaintenanceRegressionTests(VisualizerStudyTestCase):
    def remove_and_check(self, kind, identifier, expected):
        maintenance = StudyMaintenance(self.model)
        impact = maintenance.preview_removal(kind, identifier)
        unrelated = {path: path.read_bytes() for path in self.store.root.rglob('*.npy') if str(path.relative_to(self.root)) not in impact.artifact_paths}
        archive = maintenance.archive_and_remove(impact)
        self.assertEqual(list(self.repository.counts().values()), expected)
        for relative_path in impact.artifact_paths:
            self.assertFalse(self.store.resolve(relative_path).exists())
            self.assertTrue((archive / relative_path).is_file())
        for path, content in unrelated.items():
            self.assertEqual(path.read_bytes(), content)

    def test_original_anomaly_removes_all_dependent_records(self):
        self.remove_and_check('original', 'original-anomaly', [1, 0, 0, 0, 0])
        self.assertEqual(self.repository.count_match_candidates(), 0)

    def test_original_control_preserves_anomaly_sources(self):
        self.remove_and_check('original', 'original-control', [1, 1, 2, 0, 0])

    def test_real_anomaly_removes_variants_and_hybrids(self):
        self.remove_and_check('real', 'real-0', [2, 0, 0, 0, 0])

    def test_hybrid_removal_preserves_sources_and_other_hybrid(self):
        self.remove_and_check('hybrid', 'hybrid-generated', [2, 1, 2, 1, 1])

    def test_placement_removal_removes_its_entire_hybrid(self):
        self.remove_and_check('placement', 'placement-2', [2, 1, 2, 1, 1])

    def assert_archive_rollback(self, maintenance, impact, expected_error):
        before = self.repository.counts()
        hierarchy = self.repository.hierarchy()
        payloads = {path: path.read_bytes() for path in self.store.root.rglob('*.npy')}
        with self.assertRaises(expected_error):
            maintenance.archive_and_remove(impact)
        self.assertEqual(self.repository.counts(), before)
        self.assertEqual(self.repository.hierarchy(), hierarchy)
        self.assertEqual({path: path.read_bytes() for path in self.store.root.rglob('*.npy')}, payloads)

    def test_late_move_failure_restores_already_archived_files(self):
        maintenance = StudyMaintenance(self.model)
        impact = maintenance.preview_removal('real', 'real-0')
        move = shutil.move
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('archive interrupted')
            return move(*args, **kwargs)

        with patch('hybrid_sample_generator.visualization.maintenance.shutil.move', side_effect=fail_second):
            self.assert_archive_rollback(maintenance, impact, OSError)

    def test_database_failure_restores_files_and_deleted_records(self):
        maintenance = StudyMaintenance(self.model)
        impact = maintenance.preview_removal('real', 'real-0')
        with self.repository.connection() as connection:
            connection.execute("CREATE TRIGGER reject_removal BEFORE DELETE ON synthetic_anomalies BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
        self.assert_archive_rollback(maintenance, impact, sqlite3.IntegrityError)
