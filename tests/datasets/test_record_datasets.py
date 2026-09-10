import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hybrid_sample_generator.datasets.record_datasets import (
    HybridSampleDataset,
    OriginalSampleDataset,
    RealAnomalyDataset,
    SyntheticAnomalyDataset,
    save_numpy_as_npy,
)
from tests.datasets.support import DatasetStudyTestCase


class RecordDatasetTests(DatasetStudyTestCase):
    def test_original_dataset_filters_and_handles_missing_mask(self):
        controls = OriginalSampleDataset(
            self.repository,
            self.store,
            return_artifacts=("record", "ori_mask"),
            has_anomaly=False,
            is_annotated=False,
            numpy_mode=True,
        )

        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["record"].id, "original-control")
        self.assertIsNone(controls[0]["ori_mask"])

    def test_real_dataset_returns_tensors_applies_transform_and_loads_by_id(self):
        dataset = RealAnomalyDataset(
            self.repository,
            self.store,
            return_artifacts=("img", "anomaly_meta", "real_anomaly_id"),
            dtype=torch.float64,
            transform=lambda image: image + 1,
        )

        sample = dataset.load_sample_by_id("real")
        self.assertIsInstance(sample["img"], torch.Tensor)
        self.assertEqual(sample["img"].dtype, torch.float64)
        self.assertEqual(sample["real_anomaly_id"], "real")
        self.assertEqual(sample["anomaly_meta"], {"label": 1})
        with self.assertRaisesRegex(KeyError, "Record not found"):
            dataset.load_sample_by_id("missing")

    def test_synthetic_dataset_joins_real_anomaly_artifacts(self):
        dataset = SyntheticAnomalyDataset(
            self.repository,
            self.store,
            return_artifacts=(
                "synth_anomaly",
                "tgt_mask",
                "anomaly_roi",
                "anomaly_meta",
                "real_anomaly_id",
            ),
            load_to_ram=True,
            numpy_mode=True,
        )

        sample = dataset[0]
        self.assertIsInstance(sample["synth_anomaly"], np.ndarray)
        self.assertEqual(sample["real_anomaly_id"], "real")
        self.assertEqual(sample["anomaly_meta"], {"label": 1})
        np.testing.assert_array_equal(
            sample["anomaly_roi"],
            self.store.load_array(
                self.repository.get_real_anomaly("real").roi_image_path
            ),
        )

    def test_hybrid_dataset_defaults_to_materialized_records(self):
        generated = HybridSampleDataset(
            self.repository,
            self.store,
            numpy_mode=True,
        )
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0]["hybrid_sample_id"], "hybrid-generated")

        all_records = HybridSampleDataset(
            self.repository,
            self.store,
            return_artifacts=("record",),
            status=None,
            numpy_mode=True,
        )
        with self.assertRaisesRegex(ValueError, "has not been materialized"):
            all_records.load_sample_by_id("hybrid-planned")

    def test_unknown_artifact_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown original sample artifacts"):
            OriginalSampleDataset(
                self.repository,
                self.store,
                return_artifacts=("unknown",),
            )


class NumpyExportTests(unittest.TestCase):
    def test_numpy_export_adds_suffix_and_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "nested" / "sample"
            saved = Path(save_numpy_as_npy(np.array([1, 2]), target))

            self.assertEqual(saved.suffix, ".npy")
            np.testing.assert_array_equal(np.load(saved), np.array([1, 2]))
            with self.assertRaises(FileExistsError):
                save_numpy_as_npy(np.array([3]), target)
            save_numpy_as_npy(np.array([3]), target, overwrite=True)
            np.testing.assert_array_equal(np.load(saved), np.array([3]))


if __name__ == "__main__":
    unittest.main()
