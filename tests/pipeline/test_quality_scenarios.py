"""End-to-end quality checks for data and settings beyond the basic case.

Each scenario runs extraction, planning and fusion on generated data with a
generator that returns every real anomaly unchanged, like the tests in
``test_quality_end_to_end``. The scenarios cover several anomalies per hybrid,
color images, volumes, anomalies larger than the cutout, images of different
sizes, anomalous targets, several anomaly classes and continuing a saved study.
"""

import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label

from hybrid_sample_generator.configuration.root import Configuration, load_config_file
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from tests.pipeline.test_quality_end_to_end import _IdentityGenerator, _contrast

CONTRAST = 0.3


def _ellipsoid(shape, center, radii):
    grids = np.meshgrid(*[np.arange(size) for size in shape], indexing="ij")
    return sum(((grid - c) / r) ** 2 for grid, c, r in zip(grids, center, radii)) <= 1


def _samples(shapes, *, anomalous, radius=(4, 6), channels=1, margin=12, seed=7):
    """Noise images of the given spatial shapes; the first ones contain a bright ellipsoid."""
    rng = np.random.default_rng(seed)
    samples = []
    for index, shape in enumerate(shapes):
        image = (0.5 + 0.03 * rng.standard_normal((channels, *shape))).astype(np.float32)
        mask = np.zeros((1, *shape), dtype=np.uint8)
        if index < anomalous:
            distance = radius[1] + margin   # keeps the ROI of the anomaly inside the image
            center = [rng.uniform(distance, size - distance) for size in shape]
            radii = rng.uniform(*radius, size=len(shape))
            defect = _ellipsoid(shape, center, radii)
            image[:, defect] += CONTRAST
            mask[0][defect] = 1
        samples.append(InputSample(image, mask, f"sample-{index}"))
    return samples


def _configuration(root, anomaly_size, routine="local"):
    config = Configuration("scenario-study", study_folder=str(Path(root) / "study"))
    config.extraction.anomaly_size = anomaly_size
    config.extraction.add_background_noise = False
    config.generation.variants_per_real_anomaly = 1
    config.matching.routine = routine
    return config


def _run(config, samples, *, stages=("ingest", "extract", "generate", "plan", "fuse")):
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        generator = HybridDataGenerator(config, generator_model=_IdentityGenerator())
        if "ingest" in stages:
            generator.ingest_dataset(samples)
        if "extract" in stages:
            generator.extract_anomalies()
        if "generate" in stages:
            generator.generate_synthetic_anomalies()
        if "plan" in stages:
            generator.plan_hybrid_samples()
        if "fuse" in stages:
            generator.materialize_hybrid_samples()
    return generator


class QualityScenarioTests(unittest.TestCase):
    def _arrays(self, generator, hybrid):
        store, repository = generator.artifact_store, generator.repository
        load = lambda path: np.asarray(store.load_array(path))
        control = load(repository.get_original_sample(hybrid.original_sample_id).image_path)
        return control, load(hybrid.image_path), load(hybrid.segmentation_path)

    def _source_mask(self, generator, placement):
        repository = generator.repository
        synthetic = repository.get_synthetic_anomaly(placement.synthetic_anomaly_id)
        real = repository.get_real_anomaly(synthetic.real_anomaly_id)
        source = repository.get_original_sample(real.original_sample_id)
        return real, np.asarray(generator.artifact_store.load_array(source.segmentation_path))[0] > 0

    def test_two_anomalies_per_hybrid_are_separate_and_keep_their_contrast(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root, (1, 16, 16))
            config.matching.anomalies_per_hybrid = 2
            config.matching.hybrids_per_original = 2
            generator = _run(config, _samples([(96, 96)] * 9, anomalous=6))
            repository = generator.repository

            hybrids = repository.list_hybrid_samples()
            self.assertEqual(len(hybrids), 6)
            for hybrid in hybrids:
                self.assertEqual(hybrid.status, "generated")
                placements = repository.list_placements(hybrid.id)
                self.assertEqual([item.order_index for item in placements], [0, 1])
                parents = {
                    repository.get_synthetic_anomaly(item.synthetic_anomaly_id).real_anomaly_id
                    for item in placements
                }
                self.assertEqual(len(parents), 2)

                control, image, segmentation = self._arrays(generator, hybrid)
                components, count = label(segmentation[0] > 0)
                self.assertEqual(count, 2)
                for index in (1, 2):
                    component = components == index
                    self.assertGreaterEqual(_contrast(image[0], component), 0.85 * CONTRAST)
                untouched = ~binary_dilation(segmentation[0] > 0, iterations=2)
                np.testing.assert_array_equal(image[0][untouched], control[0][untouched])

    def test_color_images_keep_all_channels(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root, (3, 16, 16))
            generator = _run(config, _samples([(64, 64)] * 8, anomalous=5, channels=3))
            hybrids = generator.repository.list_hybrid_samples()
            self.assertEqual(len(hybrids), 3)
            for hybrid in hybrids:
                self.assertEqual(hybrid.status, "generated")
                control, image, segmentation = self._arrays(generator, hybrid)
                self.assertEqual(image.shape, (3, 64, 64))
                self.assertEqual(segmentation.shape, (3, 64, 64))
                mask = segmentation[0] > 0
                np.testing.assert_array_equal(segmentation[1], segmentation[0])
                for channel in range(3):
                    self.assertGreaterEqual(_contrast(image[channel], mask), 0.85 * CONTRAST)
                untouched = ~binary_dilation(mask, iterations=2)
                np.testing.assert_array_equal(image[:, untouched], control[:, untouched])

    def test_volumes_keep_position_size_and_contrast(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root, (1, 8, 8, 8))
            config.model.set_model("cVAE_ConvNeXt_3D")
            samples = _samples([(28, 28, 28)] * 6, anomalous=4, radius=(2.5, 3.5), margin=7)
            generator = _run(config, samples)
            repository = generator.repository
            hybrids = repository.list_hybrid_samples()
            self.assertEqual(len(hybrids), 2)
            for hybrid in hybrids:
                self.assertEqual(hybrid.status, "generated")
                placement = repository.list_placements(hybrid.id)[0]
                self.assertEqual(placement.spatial_dimensions, 3)
                self.assertIsNotNone(placement.position_z)
                control, image, segmentation = self._arrays(generator, hybrid)
                mask = segmentation[0] > 0
                _, source_mask = self._source_mask(generator, placement)
                self.assertAlmostEqual(mask.sum() / source_mask.sum(), 1.0, delta=0.25)
                center = np.argwhere(mask).mean(axis=0) / 28
                np.testing.assert_allclose(center, placement.position, atol=1.5 / 28)
                ring = binary_dilation(mask, iterations=4) & ~binary_dilation(mask, iterations=1)
                contrast = float(np.median(image[0][mask]) - np.median(image[0][ring]))
                self.assertGreaterEqual(contrast, 0.8 * CONTRAST)
                untouched = ~binary_dilation(mask, iterations=2)
                np.testing.assert_array_equal(image[0][untouched], control[0][untouched])

    def test_anomalies_larger_than_the_cutout_return_to_their_size(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root, (1, 16, 16))
            samples = _samples([(96, 96)] * 8, anomalous=5, radius=(11, 14))
            generator = _run(config, samples)
            repository = generator.repository
            for real in repository.list_real_anomalies():
                # The anomaly spans 22-28 px and is shrunk into the 16 px cutout.
                self.assertLess(max(real.metadata["scale_factor"]), 0.8)
            hybrids = repository.list_hybrid_samples()
            self.assertEqual(len(hybrids), 3)
            for hybrid in hybrids:
                self.assertEqual(hybrid.status, "generated")
                placement = repository.list_placements(hybrid.id)[0]
                _, image, segmentation = self._arrays(generator, hybrid)
                mask = segmentation[0] > 0
                _, source_mask = self._source_mask(generator, placement)
                self.assertAlmostEqual(mask.sum() / source_mask.sum(), 1.0, delta=0.2)
                for axis in (0, 1):
                    extent = np.ptp(np.where(mask.any(axis=1 - axis))[0]) + 1
                    source_extent = np.ptp(np.where(source_mask.any(axis=1 - axis))[0]) + 1
                    self.assertAlmostEqual(extent / source_extent, 1.0, delta=0.15)
                self.assertGreaterEqual(_contrast(image[0], mask), 0.85 * CONTRAST)

    def test_images_of_different_sizes_are_combined(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root, (1, 16, 16))
            shapes = [(64, 64)] * 4 + [(48, 80), (80, 48), (72, 72)]
            generator = _run(config, _samples(shapes, anomalous=4))
            repository = generator.repository
            hybrids = repository.list_hybrid_samples()
            self.assertEqual(len(hybrids), 3)
            seen = set()
            for hybrid in hybrids:
                self.assertEqual(hybrid.status, "generated")
                control, image, segmentation = self._arrays(generator, hybrid)
                self.assertEqual(image.shape, control.shape)
                self.assertEqual(segmentation.shape, control.shape)
                seen.add(image.shape[1:])
                mask = segmentation[0] > 0
                placement = repository.list_placements(hybrid.id)[0]
                center = np.argwhere(mask).mean(axis=0) / np.asarray(mask.shape)
                np.testing.assert_allclose(center, placement.position, atol=0.04)
                self.assertGreaterEqual(_contrast(image[0], mask), 0.85 * CONTRAST)
            self.assertEqual(seen, {(48, 80), (80, 48), (72, 72)})

    def test_anomaly_fusion_routine_targets_the_anomalous_originals(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root, (1, 16, 16), "fixed_from_extraction_anomaly_fusion")
            config.matching.hybrids_per_original = 2
            generator = _run(config, _samples([(64, 64)] * 6, anomalous=4))
            repository = generator.repository
            hybrids = repository.list_hybrid_samples()
            self.assertEqual(len(hybrids), 8)
            for hybrid in hybrids:
                self.assertEqual(hybrid.status, "generated")
                target = repository.get_original_sample(hybrid.original_sample_id)
                self.assertTrue(target.has_anomaly)
                placement = repository.list_placements(hybrid.id)[0]
                real, source_mask = self._source_mask(generator, placement)
                # The variant returns to the original of its real anomaly, at its position.
                self.assertEqual(real.original_sample_id, target.id)
                self.assertEqual(placement.position, real.source_position)
                self.assertIsNone(placement.score)
                _, _, segmentation = self._arrays(generator, hybrid)
                mask = segmentation[0] > 0
                overlap = (mask & source_mask).sum() / (mask | source_mask).sum()
                self.assertGreater(overlap, 0.6)

    def test_saved_study_continues_with_the_same_result(self):
        samples = _samples([(64, 64)] * 7, anomalous=4)
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as reference_root:
            reference = _run(_configuration(reference_root, (1, 16, 16), "batchwise"), samples)

            config = _configuration(root, (1, 16, 16), "batchwise")
            first = _run(config, samples, stages=("ingest", "extract", "generate"))
            saved = first.config.save_config_file()
            del first

            # A later session only knows the saved configuration.
            continued = _run(load_config_file(saved), None, stages=("plan", "fuse"))

            expected = reference.repository.list_placements()
            actual = continued.repository.list_placements()
            self.assertEqual(
                [(item.id, item.synthetic_anomaly_id, item.position, item.score) for item in actual],
                [(item.id, item.synthetic_anomaly_id, item.position, item.score) for item in expected],
            )
            for mine, theirs in zip(
                continued.datasets.hybrid_samples(numpy_mode=True),
                reference.datasets.hybrid_samples(numpy_mode=True),
            ):
                self.assertEqual(mine["hybrid_sample_id"], theirs["hybrid_sample_id"])
                np.testing.assert_array_equal(mine["img"], theirs["img"])
                np.testing.assert_array_equal(mine["segmentation"], theirs["segmentation"])


class MultiClassScenarioTests(unittest.TestCase):
    """Anomalies whose mask contains two classes, 1 (bright) and 2 (dark)."""

    @staticmethod
    def _samples():
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

    def _run(self, root, backend):
        config = _configuration(root, (1, 16, 16))
        config.fusion.set_backend(backend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return _run(config, self._samples())

    def test_both_classes_reach_the_hybrid_as_one_anomaly(self):
        for backend in ("classical", "poisson"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as root:
                generator = self._run(root, backend)
                repository, store = generator.repository, generator.artifact_store
                real = repository.list_real_anomalies()
                self.assertEqual(len(real), 3)
                for record in real:
                    cutout_mask = np.asarray(store.load_array(record.segmentation_path))
                    self.assertEqual(set(np.unique(cutout_mask)), {0, 1, 2})
                    self.assertEqual(record.metadata["label"], 2.0)
                hybrids = repository.list_hybrid_samples()
                self.assertEqual(len(hybrids), 3)
                for hybrid in hybrids:
                    self.assertEqual(hybrid.status, "generated")
                    image = np.asarray(store.load_array(hybrid.image_path))[0]
                    mask = np.asarray(store.load_array(hybrid.segmentation_path))[0] > 0
                    self.assertAlmostEqual(int(mask.sum()), 120, delta=12)
                    rows, columns = np.where(mask)
                    left = image[rows.min():rows.max() + 1, columns.min():columns.min() + 5]
                    right = image[rows.min():rows.max() + 1, columns.max() - 4:columns.max() + 1]
                    # The bright class stays left of the dark class.
                    self.assertGreater(float(np.median(left) - np.median(right)), 0.3)

    def test_poisson_fusion_keeps_the_class_labels(self):
        with tempfile.TemporaryDirectory() as root:
            generator = self._run(root, "poisson")
            for hybrid in generator.repository.list_hybrid_samples():
                mask = np.asarray(generator.artifact_store.load_array(hybrid.segmentation_path))
                self.assertEqual(set(np.unique(mask)), {0, 1, 2})
                self.assertAlmostEqual(int((mask == 1).sum()), 60, delta=8)
                self.assertAlmostEqual(int((mask == 2).sum()), 60, delta=8)

    @unittest.expectedFailure
    def test_classical_fusion_keeps_the_class_labels(self):
        # Known gap: classical fusion writes a binary mask, so class 2 becomes 1.
        # Remove the decorator when the labels are carried into the hybrid mask.
        with tempfile.TemporaryDirectory() as root:
            generator = self._run(root, "classical")
            for hybrid in generator.repository.list_hybrid_samples():
                mask = np.asarray(generator.artifact_store.load_array(hybrid.segmentation_path))
                self.assertEqual(set(np.unique(mask)), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
