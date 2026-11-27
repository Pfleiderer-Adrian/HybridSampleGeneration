"""End-to-end runs with every registered generator model, trained on generated data."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator

from hybrid_sample_generator.generation.registry import get_model_spec, registered_model_names

CONVNEXT = {
    "n_res_blocks": 1,
    "n_levels": 1,
    "z_channels": 4,
    "bottleneck_dim": 3,
    "drop_path_rate": 0.0,
    "dropout": 0.0,
}
SMALL_MODEL = {**CONVNEXT, "n_spade_blocks": 1}
# Small parameter sets per architecture; the test below runs every registered model.
SMALL_PARAMETERS = {
    "VAE_ResNet": {
        "n_res_blocks": 1,
        "n_levels": 1,
        "z_channels": 4,
        "bottleneck_dim": 3,
        "use_multires_skips": False,
    },
    "VAE_ConvNeXt": CONVNEXT,
    "cVAE_ConvNeXt": SMALL_MODEL,
    "paired_cVAE_ConvNeXt": SMALL_MODEL,
}


def _samples(spatial_dims=2):
    """Four anomalous samples with a bright box and two controls."""
    rng = np.random.default_rng(3)
    size = 24 if spatial_dims == 2 else 12
    shape = (1, *((size,) * spatial_dims))
    samples = []
    for index in range(6):
        image = (0.3 + 0.02 * rng.standard_normal(shape)).astype(np.float32)
        mask = np.zeros(shape, dtype=np.uint8)
        if index < 4:
            start = size // 4 + index
            box = (0, *((slice(start, start + 3),) * spatial_dims))
            image[box] += 0.5
            mask[box] = 1
        samples.append(InputSample(image, mask, f"sample-{index}"))
    return samples


class TrainedGeneratorPipelineTests(unittest.TestCase):
    def test_every_registered_model_has_small_test_parameters(self):
        architectures = {name.rsplit("_", 1)[0] for name in registered_model_names()}
        self.assertEqual(architectures, set(SMALL_PARAMETERS))

    def test_every_registered_model_trains_reloads_generates_and_fuses(self):
        for model_name in sorted(registered_model_names()):
            spec = get_model_spec(model_name)
            spatial_dims = spec.spatial_dims
            cutout = (1, *((8 if spatial_dims == 2 else 4,) * spatial_dims))
            with (
                self.subTest(model=model_name),
                tempfile.TemporaryDirectory() as root,
                patch("torch.cuda.is_available", return_value=False),
            ):
                config = Configuration("model-study", study_folder=str(Path(root) / "study"))
                config.extraction.anomaly_size = cutout
                config.extraction.add_background_noise = False
                config.model.set_model(model_name)
                for name, value in SMALL_PARAMETERS[model_name.rsplit("_", 1)[0]].items():
                    setattr(config.model.parameters, name, value)
                config.model.search.clear()
                config.training.num_trials = 1
                config.training.epochs = 2
                config.training.batch_size = 2
                config.training.validation_ratio = 0.25
                config.generation.variants_per_real_anomaly = 2
                config.matching.routine = "local"

                silent = io.StringIO()
                with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
                    generator = HybridDataGenerator(config)
                    generator.ingest_dataset(_samples(spatial_dims))
                    real = generator.extract_anomalies()
                    generator.train_generator()
                    # A new facade loads the trained model from the study folder.
                    restarted = HybridDataGenerator(config)
                    restarted.load_generator()
                    synthetic = restarted.generate_synthetic_anomalies()
                    restarted.plan_hybrid_samples()
                    hybrids = restarted.materialize_hybrid_samples()

                self.assertEqual(len(real), 4)
                self.assertEqual(len(synthetic), 8)
                load = lambda path: np.asarray(restarted.artifact_store.load_array(path))
                changed_masks = 0
                for record in synthetic:
                    parent = restarted.repository.get_real_anomaly(record.real_anomaly_id)
                    image, mask = load(record.image_path), load(record.segmentation_path)
                    self.assertEqual(image.shape, cutout)
                    self.assertEqual(mask.shape, cutout)
                    self.assertTrue(np.isfinite(image).all())
                    self.assertGreater(int(np.count_nonzero(mask)), 0)
                    changed_masks += not np.array_equal(mask, load(parent.segmentation_path))
                if spec.uses_masks:
                    # The default mask transforms give the variants a new target mask.
                    self.assertGreater(changed_masks, 0)

                self.assertEqual(len(hybrids), 2)
                image_shape = _samples(spatial_dims)[0].image.shape
                for hybrid in hybrids:
                    self.assertEqual(hybrid.status, "generated")
                    image = load(hybrid.image_path)
                    self.assertEqual(image.shape, image_shape)
                    self.assertEqual(load(hybrid.segmentation_path).shape, image_shape)
                    self.assertTrue(np.isfinite(image).all())

    def test_generation_is_reproducible_for_a_loaded_model(self):
        with tempfile.TemporaryDirectory() as root, patch("torch.cuda.is_available", return_value=False):
            config = Configuration("repeat-study", study_folder=str(Path(root) / "study"))
            config.extraction.anomaly_size = (1, 8, 8)
            config.extraction.add_background_noise = False
            config.model.set_model("cVAE_ConvNeXt_2D")
            for name, value in SMALL_MODEL.items():
                setattr(config.model.parameters, name, value)
            config.model.search.clear()
            config.training.num_trials = 1
            config.training.epochs = 1
            config.training.batch_size = 2
            config.generation.variants_per_real_anomaly = 2

            silent = io.StringIO()
            runs = []
            with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
                generator = HybridDataGenerator(config)
                generator.ingest_dataset(_samples())
                generator.extract_anomalies()
                generator.train_generator()
                for _ in range(2):
                    records = generator.generate_synthetic_anomalies()
                    runs.append([
                        (
                            record.id,
                            record.seed,
                            np.asarray(generator.artifact_store.load_array(record.image_path)).copy(),
                            np.asarray(generator.artifact_store.load_array(record.segmentation_path)).copy(),
                        )
                        for record in records
                    ])

            self.assertEqual(len(runs[0]), 8)
            self.assertEqual(len({seed for _, seed, _, _ in runs[0]}), 8)
            for first, second in zip(*runs):
                self.assertEqual(first[:2], second[:2])
                np.testing.assert_array_equal(first[2], second[2])
                np.testing.assert_array_equal(first[3], second[3])


if __name__ == "__main__":
    unittest.main()
