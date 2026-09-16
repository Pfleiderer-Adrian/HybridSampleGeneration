"""Persisted split identity, source validation and holdout separation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.mvtec_ad2.splits import SplitConfiguration, load_or_create_manifest

from .test_downstream import create_images


class SplitTests(unittest.TestCase):
    def test_reuses_saved_split_and_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            path = root / "study" / "split_manifest.json"
            manifest = load_or_create_manifest(path, root / "can")
            with patch("examples.mvtec_ad2.splits.create_manifest", side_effect=AssertionError("must reuse saved split")):
                self.assertEqual(manifest, load_or_create_manifest(path, root / "can"))
            with self.assertRaisesRegex(ValueError, "settings differ"):
                load_or_create_manifest(path, root / "can", SplitConfiguration(seed=7))

    def test_rejects_modified_manifest_and_missing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            path = root / "split_manifest.json"
            manifest = load_or_create_manifest(path, root / "can")
            modified = {**manifest, "fingerprint": "changed"}
            path.write_text(json.dumps(modified))
            with self.assertRaisesRegex(ValueError, "modified"):
                load_or_create_manifest(path, root / "can")
            path.write_text(json.dumps(manifest))
            Path(manifest["partitions"]["train"][0]["image_path"]).unlink()
            with self.assertRaises(FileNotFoundError):
                load_or_create_manifest(path, root / "can")

    def test_zero_test_fraction_keeps_train_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            manifest = load_or_create_manifest(root / "split.json", root / "can", SplitConfiguration(test_fraction=0))
            self.assertFalse(manifest["partitions"]["test"])
            self.assertTrue(manifest["partitions"]["train"])
            self.assertTrue(manifest["partitions"]["validation"])
