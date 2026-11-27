"""End-to-end quality checks on generated test data.

The other pipeline tests assert that the stages run and keep their records
consistent. These tests assert properties of the produced hybrids: where the
anomaly is, how large its mask is, how strongly it differs from its
surroundings and that nothing else in the image changed.

The generator model is replaced by one that returns each real anomaly
unchanged. Every difference between a hybrid and its real anomaly is then
caused by extraction, planning or fusion.
"""

import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.evaluation.service import evaluate_study
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator

IMAGE_SIZE = 64
DEFECT_CONTRAST = 0.3
ANOMALOUS_SAMPLES = 6
CONTROL_SAMPLES = 4


class _IdentityGenerator:
    """Return the real anomaly cutout and its mask as the synthetic variant."""

    def generate(self, sample, **_kwargs):
        return (
            np.asarray(sample["img"], dtype=np.float32),
            np.asarray(sample["ori_mask"], dtype=np.uint8),
        )


def _samples(*, at_right_border=False):
    """Textured images; the anomalous ones contain one bright ellipse.

    The ellipse lies away from the border, or is cut off by the right border.
    """
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    samples = []
    for index in range(ANOMALOUS_SAMPLES + CONTROL_SAMPLES):
        image = (0.5 + 0.03 * rng.standard_normal((1, IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32)
        mask = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        if index < ANOMALOUS_SAMPLES:
            center_y, center_x = rng.uniform(22, IMAGE_SIZE - 22, size=2)
            radius_y, radius_x = rng.uniform(4, 6, size=2)
            if at_right_border:
                center_x = IMAGE_SIZE - 2
            defect = ((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2 <= 1
            image[0][defect] += DEFECT_CONTRAST
            mask[0][defect] = 1
            name = f"anomaly-{index}"
        else:
            name = f"control-{index}"
        samples.append(InputSample(image, mask, name))
    return samples


def _contrast(image, mask):
    """Median intensity inside the mask minus the median in a ring around it."""
    ring = binary_dilation(mask, iterations=5) & ~binary_dilation(mask, iterations=2)
    return float(np.median(image[mask]) - np.median(image[ring]))


def _configuration(root, routine, backend="classical"):
    config = Configuration("quality-study", study_folder=str(Path(root) / "study"))
    config.extraction.anomaly_size = (1, 16, 16)
    config.extraction.add_background_noise = False
    config.generation.variants_per_real_anomaly = 1
    config.matching.routine = routine
    config.matching.hybrids_per_original = 2
    # Fusion runs with the package defaults, so a default that weakens or
    # displaces fused anomalies fails these tests.
    config.fusion.set_backend(backend)
    return config


def _run(config, *, samples=None, generator_model=None, raise_on_error=True):
    """Run all stages up to materialization and return the generator."""
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        generator = HybridDataGenerator(
            config, generator_model=generator_model or _IdentityGenerator()
        )
        generator.ingest_dataset(_samples() if samples is None else samples)
        generator.extract_anomalies()
        generator.generate_synthetic_anomalies()
        generator.plan_hybrid_samples()
        generator.materialize_hybrid_samples(raise_on_error=raise_on_error)
    return generator


class HybridQualityTests(unittest.TestCase):
    def _assert_hybrid_quality(self, generator, routine, *, minimum_contrast_share):
        repository, store = generator.repository, generator.artifact_store
        load = lambda path: np.asarray(store.load_array(path))
        hybrids = repository.list_hybrid_samples()
        self.assertEqual(len(hybrids), CONTROL_SAMPLES * 2)
        for hybrid in hybrids:
            with self.subTest(routine=routine, hybrid=hybrid.id):
                self.assertEqual(hybrid.status, "generated")
                control_record = repository.get_original_sample(hybrid.original_sample_id)
                self.assertFalse(control_record.has_anomaly)
                control = load(control_record.image_path)[0]
                image = load(hybrid.image_path)[0]
                mask = load(hybrid.segmentation_path)[0] > 0

                placements = repository.list_placements(hybrid.id)
                self.assertEqual(len(placements), 1)
                placement = placements[0]
                self.assertEqual(placement.method, routine)
                synthetic = repository.get_synthetic_anomaly(placement.synthetic_anomaly_id)
                real = repository.get_real_anomaly(synthetic.real_anomaly_id)
                source = repository.get_original_sample(real.original_sample_id)
                source_mask = load(source.segmentation_path)[0] > 0

                # The mask has the size of the real anomaly and sits at the planned position.
                self.assertTrue(mask.any())
                self.assertAlmostEqual(mask.sum() / source_mask.sum(), 1.0, delta=0.15)
                center = np.argwhere(mask).mean(axis=0) / IMAGE_SIZE
                np.testing.assert_allclose(center, placement.position, atol=1.5 / IMAGE_SIZE)

                # The anomaly keeps its contrast to the surroundings.
                self.assertGreaterEqual(
                    _contrast(image, mask), minimum_contrast_share * DEFECT_CONTRAST
                )
                # Nothing outside the anomaly and its blending margin changed.
                untouched = ~binary_dilation(mask, iterations=2)
                np.testing.assert_array_equal(image[untouched], control[untouched])
                self.assertTrue(np.isfinite(image).all())

    def test_template_matching_routines_produce_faithful_hybrids(self):
        for routine in ("local", "global", "batchwise"):
            with self.subTest(routine=routine), tempfile.TemporaryDirectory() as root:
                generator = _run(_configuration(root, routine))
                self._assert_hybrid_quality(generator, routine, minimum_contrast_share=0.9)
                for placement in generator.repository.list_placements():
                    self.assertIsNotNone(placement.score)
                    self.assertGreaterEqual(placement.score, -1.0)
                    self.assertLessEqual(placement.score, 1.0)

    def test_fixed_routine_reuses_the_source_position(self):
        with tempfile.TemporaryDirectory() as root:
            routine = "fixed_from_extraction_control_fusion"
            generator = _run(_configuration(root, routine))
            self._assert_hybrid_quality(generator, routine, minimum_contrast_share=0.9)
            repository = generator.repository
            for placement in repository.list_placements():
                synthetic = repository.get_synthetic_anomaly(placement.synthetic_anomaly_id)
                real = repository.get_real_anomaly(synthetic.real_anomaly_id)
                self.assertIsNone(placement.score)
                self.assertEqual(placement.position, real.source_position)

    def test_anomalies_from_the_border_are_placed_at_the_border(self):
        # The ROI of an anomaly at the border is shifted inside the image, so the
        # anomaly is not in the ROI center. Planning has to place it where it sat
        # inside the ROI. Matching by distance only looks for the source position.
        for routine in ("local", "fixed_from_extraction_control_fusion"):
            with self.subTest(routine=routine), tempfile.TemporaryDirectory() as root:
                config = _configuration(root, routine)
                config.matching.intensity_weight = 0.0
                config.matching.gradient_weight = 0.0
                config.matching.distance_weight = 1.0
                generator = _run(config, samples=_samples(at_right_border=True))
                repository, store = generator.repository, generator.artifact_store
                if routine == "local":
                    offsets = [record.metadata["roi_center_offset"] for record in repository.list_real_anomalies()]
                    self.assertTrue(all(offset[1] > 2 for offset in offsets))

                hybrids = repository.list_hybrid_samples(status="generated")
                self.assertEqual(len(hybrids), CONTROL_SAMPLES * 2)
                for hybrid in hybrids:
                    mask = np.asarray(store.load_array(hybrid.segmentation_path))[0] > 0
                    columns = np.where(mask.any(axis=0))[0]
                    # Positions are rounded to whole pixels, which can move the
                    # anomaly by one pixel. Without the stored offset it would end
                    # several pixels before the border.
                    self.assertGreaterEqual(int(columns.max()), IMAGE_SIZE - 2)
                    placement = repository.list_placements(hybrid.id)[0]
                    synthetic = repository.get_synthetic_anomaly(placement.synthetic_anomaly_id)
                    real = repository.get_real_anomaly(synthetic.real_anomaly_id)
                    np.testing.assert_allclose(
                        placement.position, real.source_position, atol=1.5 / IMAGE_SIZE
                    )

    def test_masks_that_exceed_the_image_fail_with_an_explanation(self):
        class WholeCutoutMask:
            """Mark the whole cutout as anomaly, as a far too low threshold does."""

            def generate(self, sample, **_kwargs):
                cutout = np.asarray(sample["img"], dtype=np.float32)
                return cutout, np.ones_like(cutout, dtype=np.uint8)

        # Tall anomalies are shrunk to fit the 16 px high cutout. A mask that fills
        # the 32 px wide cutout is restored to about twice the image width.
        yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
        rng = np.random.default_rng(11)
        samples = []
        for index in range(4):
            image = (0.5 + 0.03 * rng.standard_normal((1, IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32)
            mask = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            if index < 2:
                defect = (np.abs(yy - 32) <= 30) & (np.abs(xx - 32) <= 3)
                image[0][defect] += DEFECT_CONTRAST
                mask[0][defect] = 1
            samples.append(InputSample(image, mask, f"sample-{index}"))

        with tempfile.TemporaryDirectory() as root, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            config = _configuration(root, "fixed_from_extraction_control_fusion")
            config.extraction.anomaly_size = (1, 16, 32)
            config.model.set_model("VAE_ResNet_2D")
            config.matching.hybrids_per_original = 1
            generator = _run(
                config, samples=samples, generator_model=WholeCutoutMask(), raise_on_error=False,
            )
            hybrids = generator.repository.list_hybrid_samples()
            self.assertEqual(len(hybrids), 2)
            for hybrid in hybrids:
                self.assertEqual(hybrid.status, "failed")
                self.assertIn("does not fit into the control image", hybrid.error)
                self.assertIsNone(hybrid.image_path)
            # The default call reports the failures instead of returning silently.
            with self.assertRaisesRegex(RuntimeError, "Failed to materialize 2 hybrid samples"):
                silent = io.StringIO()
                with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
                    generator.materialize_hybrid_samples()

    def test_generation_reports_masks_far_larger_than_the_real_anomaly(self):
        class WholeCutoutMask:
            def generate(self, sample, **_kwargs):
                cutout = np.asarray(sample["img"], dtype=np.float32)
                return cutout, np.ones_like(cutout, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root, "local")
            config.model.set_model("VAE_ResNet_2D")
            silent = io.StringIO()
            with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
                generator = HybridDataGenerator(config, generator_model=WholeCutoutMask())
                generator.ingest_dataset(_samples())
                generator.extract_anomalies()
                with self.assertWarnsRegex(RuntimeWarning, "synthetic anomalies have a mask more than"):
                    generator.generate_synthetic_anomalies()

    def test_poisson_fusion_produces_faithful_hybrids(self):
        with tempfile.TemporaryDirectory() as root, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            routine = "fixed_from_extraction_control_fusion"
            generator = _run(_configuration(root, routine, backend="poisson"))
            self._assert_hybrid_quality(generator, routine, minimum_contrast_share=0.7)

    def test_planning_and_fusion_are_reproducible(self):
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            results = []
            for root in (first_root, second_root):
                generator = _run(_configuration(root, "batchwise"))
                placements = [
                    (item.hybrid_sample_id, item.synthetic_anomaly_id, item.position, item.score)
                    for item in generator.repository.list_placements()
                ]
                images = [
                    np.asarray(sample["img"])
                    for sample in generator.datasets.hybrid_samples(numpy_mode=True)
                ]
                results.append((placements, images))
            self.assertEqual(results[0][0], results[1][0])
            for first, second in zip(results[0][1], results[1][1]):
                np.testing.assert_array_equal(first, second)

    def test_postprocessing_evaluation_and_datasets_on_a_finished_study(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root, "local")
            config.postprocessing.enabled = True
            config.postprocessing.rules.mask_area_fraction.enabled = True
            config.postprocessing.rules.mask_area_fraction.minimum = 0.005
            config.postprocessing.rules.mask_area_fraction.maximum = 0.05
            generator = _run(config)
            hybrid_count = CONTROL_SAMPLES * 2
            silent = io.StringIO()
            with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
                summary = generator.postprocess_hybrid_samples()
                # Every anomaly covers 1-3 % of the image and passes the area rule.
                self.assertEqual(summary["total"], hybrid_count)
                self.assertEqual(summary["passed"], hybrid_count)
                self.assertEqual(summary["failed"], 0)

                # A stricter rule rejects every hybrid without touching the images.
                config.postprocessing.rules.mask_area_fraction.maximum = 0.006
                rejected = HybridDataGenerator(config).postprocess_hybrid_samples()
                self.assertEqual(rejected["failed"], hybrid_count)
                self.assertEqual(len(generator.repository.list_hybrid_samples(status="generated")), hybrid_count)

                results = evaluate_study(config)

            # An unchanged variant has no texture or shape difference to its parent.
            for comparison in ("glcm_cutout", "volume_cutout"):
                self.assertEqual(results[comparison]["sample_counter"], ANOMALOUS_SAMPLES)
                for differences in results[comparison]["all_diffs"].values():
                    np.testing.assert_allclose(differences, 0.0, atol=1e-9)
            self.assertEqual(results["glcm_roi"]["sample_counter"], hybrid_count)
            self.assertTrue(Path(config.study.paths.metric_diffs_csv).is_file())

            views = generator.datasets
            self.assertEqual(len(views.original_samples(has_anomaly=True)), ANOMALOUS_SAMPLES)
            self.assertEqual(len(views.original_samples(has_anomaly=False)), CONTROL_SAMPLES)
            self.assertEqual(len(views.real_anomalies()), ANOMALOUS_SAMPLES)
            self.assertEqual(len(views.synthetic_anomalies()), ANOMALOUS_SAMPLES)
            hybrids = views.hybrid_samples(numpy_mode=True)
            self.assertEqual(len(hybrids), hybrid_count)
            sample = hybrids[0]
            self.assertEqual(sample["img"].shape, (1, IMAGE_SIZE, IMAGE_SIZE))
            self.assertEqual(sample["segmentation"].shape, (1, IMAGE_SIZE, IMAGE_SIZE))
            self.assertEqual(hybrids.load_sample_by_id(sample["hybrid_sample_id"])["hybrid_sample_id"], sample["hybrid_sample_id"])


if __name__ == "__main__":
    unittest.main()
