"""Regression tests for rollback of replaced originals and database writes."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.study_repository import StudyRepository
from hybrid_sample_generator.pipeline.ingestion import ingest_dataset


class IngestionAtomicityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = ArtifactStore(self.temporary.name)
        self.repository = StudyRepository(Path(self.temporary.name) / 'study.sqlite')
        self.original = InputSample(np.zeros((1, 4, 4)), np.ones((1, 4, 4), dtype=np.uint8), 'existing')
        self.ingest([self.original])
        self.records = self.repository.list_original_samples()
        self.payloads = {path: path.read_bytes() for path in self.store.root.rglob('*.npy')}

    def ingest(self, samples):
        return ingest_dataset(samples, self.repository, self.store, expected_spatial_dimensions=2, expected_channels=1)

    def changed(self):
        return InputSample(np.ones((1, 4, 4)), np.zeros((1, 4, 4), dtype=np.uint8), 'existing')

    def assert_rolled_back(self):
        self.assertEqual(self.repository.list_original_samples(), self.records)
        self.assertEqual({path: path.read_bytes() for path in self.store.root.rglob('*.npy')}, self.payloads)
        self.assertFalse(list(Path(self.temporary.name).glob('.artifact-backup-*')))

    def test_late_validation_duplicate_and_iterator_failures_restore_old_arrays(self):
        def interrupted():
            yield self.changed()
            raise RuntimeError('source interrupted')

        cases = [([self.changed(), InputSample(np.ones((2, 4, 4)), None, 'invalid')], ValueError),
                 ([self.changed(), self.changed()], ValueError),
                 (interrupted(), RuntimeError)]
        for samples, error in cases:
            with self.subTest(error=error), self.assertRaises(error):
                self.ingest(samples)
            self.assert_rolled_back()

    def test_late_write_failure_restores_image_and_mask_and_removes_new_arrays(self):
        save = self.store.save_entity_array
        calls = 0

        def fail_fourth(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise OSError('mask write failed')
            return save(*args, **kwargs)

        with patch.object(self.store, 'save_entity_array', side_effect=fail_fourth), self.assertRaises(OSError):
            self.ingest([self.changed(), InputSample(np.ones((1, 4, 4)), np.ones((1, 4, 4)), 'new')])
        self.assert_rolled_back()

    def test_database_replacement_failure_restores_catalog_and_files(self):
        with self.repository.connection() as connection:
            connection.execute("CREATE TRIGGER reject_original BEFORE INSERT ON original_samples BEGIN SELECT RAISE(ABORT, 'injected database failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.ingest([self.changed()])
        self.assert_rolled_back()

    def test_successful_replacement_updates_both_files_and_catalog(self):
        self.ingest([self.changed()])
        record = self.repository.list_original_samples()[0]
        self.assertFalse(record.has_anomaly)
        np.testing.assert_array_equal(self.store.load_array(record.image_path), self.changed().image)
        np.testing.assert_array_equal(self.store.load_array(record.segmentation_path), self.changed().segmentation)
        self.assertFalse(list(Path(self.temporary.name).glob('.artifact-backup-*')))

    def test_database_commit_failure_restores_files_and_rolls_back_catalog(self):
        connect = sqlite3.connect

        class FailingCommitConnection(sqlite3.Connection):
            def commit(self):
                raise sqlite3.OperationalError("injected commit failure")

        def failing_connection(*args, **kwargs):
            return connect(*args, **kwargs, factory=FailingCommitConnection)

        with patch("hybrid_sample_generator.persistence.study_repository.sqlite3.connect", side_effect=failing_connection), self.assertRaisesRegex(sqlite3.OperationalError, "commit failure"):
            self.ingest([self.changed()])
        self.assert_rolled_back()
