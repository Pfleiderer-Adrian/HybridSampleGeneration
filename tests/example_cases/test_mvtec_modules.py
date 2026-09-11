"""Tests for MVTec AD 2 discovery and dataset adapters."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.mvtec_ad2.dataloader import MVTecAD2Dataloader
from examples.mvtec_ad2.discovery import discover_mvtecad2_categories
from examples.mvtec_ad2.records import MVTecAD2Sample
from examples.mvtec_ad2.runner import _segmentation_for_png


class MVTecModuleBoundaryTests(unittest.TestCase):
    def test_category_discovery_requires_expected_dataset_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            valid = root / "can"
            (valid / "train").mkdir(parents=True)
            (valid / "test_public").mkdir()
            (root / "incomplete" / "train").mkdir(parents=True)

            self.assertEqual(discover_mvtecad2_categories(root), ["can"])

    def test_dataloader_exposes_legacy_tuple_and_input_sample_views(self):
        sample = MVTecAD2Sample(
            image_path=Path("sample.png"),
            mask_path=None,
            sample_id="train_good_sample.png",
            split="train",
            label="good",
        )
        loader = MVTecAD2Dataloader([sample])

        with patch(
            "examples.mvtec_ad2.dataloader.load_image_array",
            return_value=np.ones((5, 7), dtype=np.uint8),
        ):
            image, segmentation, sample_id = next(iter(loader))
            input_sample = next(loader.iter_input_samples())

        self.assertEqual(image.shape, (1, 5, 7))
        self.assertEqual(segmentation.shape, image.shape)
        self.assertEqual(sample_id, sample.sample_id)
        self.assertEqual(input_sample.source_name, sample.sample_id)
        self.assertEqual(input_sample.metadata, {"split": "train", "label": "good"})

    def test_segmentation_export_is_binary_and_channel_first(self):
        segmentation = np.array([[[0, 2], [-1, 3]]], dtype=np.float32)
        exported = _segmentation_for_png(segmentation)

        self.assertEqual(exported.dtype, np.uint8)
        self.assertTrue(np.array_equal(exported, [[[0, 255], [0, 255]]]))


if __name__ == "__main__":
    unittest.main()
