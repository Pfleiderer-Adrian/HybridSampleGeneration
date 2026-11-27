"""Class labels of anomaly masks must reach the hybrid masks."""

import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator


class _IdentityGenerator:
    """Return the real anomaly cutout and its mask as the synthetic variant."""

    def generate(self, sample, **_kwargs):
        return (
            np.asarray(sample["img"], dtype=np.float32),
            np.asarray(sample["ori_mask"], dtype=np.uint8),
        )


def _samples():
    """Three originals with a bright class 1 next to a dark class 2, and three controls."""
    rng = np.random.default_rng(5)
    samples = []
    for index in range(6):
        image = (0.5 + 0.03 * rng.standard_normal((1, 64, 64))).astype(np.float32)
        mask = np.zeros((1, 64, 64), dtype=np.uint8)
        if index < 3:
            row, column = 18 + 6 * index, 20 + 4 * index
            image[0, row:row + 10, column:column + 6] += 0.3
            mask[0, row:row + 10, column:column + 6] = 1
            image[0, row:row + 10, column + 6:column + 12] -= 0.2
            mask[0, row:row + 10, column + 6:column + 12] = 2
        samples.append(InputSample(image, mask, f"sample-{index}"))
    return samples


class HybridClassLabelTests(unittest.TestCase):
    def test_both_backends_keep_the_class_labels(self):
        for backend in ("classical", "poisson"):
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as root,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", RuntimeWarning)
                config = Configuration("label-study", study_folder=str(Path(root) / "study"))
                config.extraction.anomaly_size = (1, 16, 16)
                config.extraction.add_background_noise = False
                config.generation.variants_per_real_anomaly = 1
                config.matching.routine = "local"
                config.fusion.set_backend(backend)

                silent = io.StringIO()
                with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
                    generator = HybridDataGenerator(config, generator_model=_IdentityGenerator())
                    generator.ingest_dataset(_samples())
                    generator.extract_anomalies()
                    generator.generate_synthetic_anomalies()
                    generator.plan_hybrid_samples()
                    hybrids = generator.materialize_hybrid_samples()

                self.assertEqual(len(hybrids), 3)
                for hybrid in hybrids:
                    mask = np.asarray(generator.artifact_store.load_array(hybrid.segmentation_path))[0]
                    image = np.asarray(generator.artifact_store.load_array(hybrid.image_path))[0]
                    self.assertEqual(set(np.unique(mask)), {0, 1, 2})
                    self.assertAlmostEqual(int((mask == 1).sum()), 60, delta=8)
                    self.assertAlmostEqual(int((mask == 2).sum()), 60, delta=8)
                    # The labels mark the matching image content: class 1 is brighter.
                    self.assertGreater(
                        float(np.median(image[mask == 1]) - np.median(image[mask == 2])), 0.3
                    )


if __name__ == "__main__":
    unittest.main()
