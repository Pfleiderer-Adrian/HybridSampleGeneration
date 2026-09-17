"""Artifact safety and atomic write regressions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_unsafe_paths_ids_and_roles_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root)
            for path in ('../outside.npy', '/tmp/outside.npy'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    store.resolve(path)
            for value in ('../x', 'x/y', 'x\\y', ''):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    store.relative_path('original_samples', value, 'image')
                with self.assertRaises(ValueError):
                    store.relative_path('original_samples', 'safe', value)

    def test_write_failures_preserve_existing_file_and_remove_temporary_files(self):
        for operation in ('numpy.save', 'hybrid_sample_generator.persistence.artifact_store.os.replace'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as root:
                store = ArtifactStore(root)
                path = store.save_entity_array('original_samples', 'sample', 'image', np.ones((1, 2, 2)))
                before = set(Path(root).rglob('*.npy'))
                with patch(operation, side_effect=OSError('disk failure')), self.assertRaises(OSError):
                    store.save_array(path, np.zeros((1, 2, 2)))
                np.testing.assert_array_equal(store.load_array(path), np.ones((1, 2, 2)))
                self.assertEqual(set(Path(root).rglob('*.npy')), before)

    def test_multi_file_rollback_restores_first_version_and_removes_new_files(self):
        with tempfile.TemporaryDirectory() as root:
            store = ArtifactStore(root)
            path = store.save_entity_array("original_samples", "sample", "image", np.ones((1, 2, 2)))
            with self.assertRaises(RuntimeError):
                with store.transaction():
                    store.save_array(path, np.zeros((1, 2, 2)))
                    store.save_array(path, np.full((1, 2, 2), 9))
                    new = store.save_entity_array("original_samples", "new", "image", np.ones((1, 2, 2)))
                    raise RuntimeError("interrupted multi-file update")
            np.testing.assert_array_equal(store.load_array(path), np.ones((1, 2, 2)))
            self.assertFalse(store.exists(new))
            with store.transaction():
                store.save_array(path, np.zeros((1, 2, 2)))
            np.testing.assert_array_equal(store.load_array(path), np.zeros((1, 2, 2)))
