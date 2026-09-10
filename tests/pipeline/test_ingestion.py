import tempfile
import unittest
from pathlib import Path

import numpy as np

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator


class PipelineIngestionTests(unittest.TestCase):
    def test_ingest_classifies_unannotated_controls(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration(
                "unannotated-control-study",
                "VAE_ResNet_2D",
                (1, 8, 8),
                study_folder=str(Path(root) / "study"),
            )
            image = np.zeros((1, 16, 16), dtype=np.float32)
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[:, 3:6, 3:6] = 1

            generator = HybridDataGenerator(config)
            summary = generator.ingest_dataset(
                [
                    InputSample(image, mask, "anomaly"),
                    InputSample(image.copy(), None, "unannotated-control"),
                ]
            )

            self.assertEqual(summary.annotated_samples, 1)
            self.assertEqual(summary.unannotated_samples, 1)
            controls = generator.datasets.original_samples(
                return_artifacts=("record",),
                has_anomaly=False,
                is_annotated=False,
                numpy_mode=True,
            )
            self.assertEqual(len(controls), 1)
            self.assertFalse(controls[0]["record"].has_anomaly)
