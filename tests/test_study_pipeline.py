import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion_backend.interfaces import FusionOutput
from synthesizer.Configuration import Configuration
from synthesizer.Evaluation import evaluate_study
from synthesizer.HybridDataGenerator import HybridDataGenerator
from synthesizer.InputSample import InputSample


class _FakeGenerator:
    def generate(self, sample, **_kwargs):
        image = np.asarray(sample["img"], dtype=np.float32)
        noise = np.random.normal(0.0, 0.01, image.shape).astype(np.float32)
        return image + noise, np.asarray(sample["ori_mask"], dtype=np.uint8)


class _FakeFusionBackend:
    def warmup(self, *_args, **_kwargs):
        return self

    def fuse(self, sample, control_img, position, **_kwargs):
        image = np.asarray(control_img).copy()
        segmentation = np.zeros_like(image, dtype=np.uint8)
        spatial_shape = image.shape[1:]
        center = [
            min(max(int(round(value * size)), 0), size - 1)
            for value, size in zip(position, spatial_shape)
        ]
        slices = tuple(slice(max(value - 1, 0), min(value + 1, size)) for value, size in zip(center, spatial_shape))
        image[(slice(None), *slices)] += 0.5
        segmentation[(slice(None), *slices)] = 1
        return FusionOutput(
            image=image,
            segmentation=segmentation,
            roi=np.asarray(sample["synth_anomaly"]),
            roi_mask=np.asarray(sample["tgt_mask"]),
        )


def _anomaly_samples():
    samples = []
    for sample_index in range(2):
        image = np.zeros((1, 32, 32), dtype=np.float32)
        segmentation = np.zeros_like(image, dtype=np.uint8)
        first = 4 + sample_index
        second = 22 - sample_index
        image[:, first:first + 4, first:first + 4] = 0.4 + sample_index * 0.1
        image[:, second:second + 4, second:second + 4] = 0.8
        segmentation[:, first:first + 4, first:first + 4] = 1
        segmentation[:, second:second + 4, second:second + 4] = 1
        samples.append(InputSample(image, segmentation, f"anomaly-{sample_index}"))
    return samples


def _control_samples():
    return [
        InputSample(
            np.full((1, 32, 32), 0.2 + index * 0.1, dtype=np.float32),
            np.zeros((1, 32, 32), dtype=np.uint8),
            f"control-{index}",
        )
        for index in range(2)
    ]


class StudyPipelineTests(unittest.TestCase):
    def test_multiple_variants_and_normalized_placements_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration(
                "normalized-study",
                "VAE_ResNet_2D",
                (1, 8, 8),
                study_folder=str(Path(root) / "study"),
            )
            config.extraction.min_coverage_ratio = 0.0
            config.extraction.add_background_noise = False
            config.extraction.normalization = None
            config.extraction.roi.fixed_size = (8, 8)
            config.generation.variants_per_real_anomaly = 3
            config.matching.routine = "fixed_from_extraction_control_fusion"
            config.matching.hybrids_per_original = 3
            config.matching.anomalies_per_hybrid = 2
            config.matching.max_anomalies_per_hybrid_deviation = 0
            config.matching.reuse_synthetic_across_hybrids = True
            config.matching.allow_sibling_variants_in_same_hybrid = False
            config.validate()

            extractor = HybridDataGenerator(config)
            real = extractor.extract_anomalies(_anomaly_samples())
            self.assertEqual(len(real), 4)

            generator = HybridDataGenerator(config, generator_model=_FakeGenerator())
            synthetic = generator.generate_synthetic_anomalies()
            self.assertEqual(len(synthetic), 12)
            self.assertEqual(
                {item.variant_index for item in synthetic}, {0, 1, 2}
            )

            planner = HybridDataGenerator(config)
            planned = planner.plan_hybrid_samples(_control_samples())
            self.assertEqual(len(planned), 6)
            placements = planner.repository.list_placements()
            self.assertEqual(len(placements), 12)
            for hybrid in planned:
                current = planner.repository.list_placements(hybrid.id)
                self.assertEqual(len(current), 2)
                real_ids = {
                    planner.repository.get_synthetic_anomaly(item.synthetic_anomaly_id).real_anomaly_id
                    for item in current
                }
                self.assertEqual(len(real_ids), 2)
                self.assertEqual(len({item.synthetic_anomaly_id for item in current}), 2)
                for item in current:
                    self.assertEqual(item.spatial_dimensions, 2)
                    self.assertIsNone(item.position_z)
                    self.assertGreaterEqual(item.position_y, 0.0)
                    self.assertLessEqual(item.position_y, 1.0)
                    self.assertGreaterEqual(item.position_x, 0.0)
                    self.assertLessEqual(item.position_x, 1.0)

            materializer = HybridDataGenerator(
                config,
                fusion_backend=_FakeFusionBackend(),
            )
            generated = materializer.materialize_hybrid_samples()
            self.assertEqual(len(generated), 6)
            artifact_paths = set()
            for hybrid in generated:
                self.assertEqual(hybrid.status, "generated")
                self.assertTrue(materializer.artifact_store.exists(hybrid.image_path))
                self.assertTrue(materializer.artifact_store.exists(hybrid.segmentation_path))
                artifact_paths.update((hybrid.image_path, hybrid.segmentation_path))
            self.assertEqual(len(artifact_paths), 12)
            self.assertTrue(
                all(
                    materializer.artifact_store.exists(item.roi_image_path)
                    for item in materializer.repository.list_placements()
                )
            )

            hybrid_dataset = materializer.datasets.hybrid_samples(
                load_to_ram=False,
                numpy_mode=True,
            )
            self.assertEqual(len(hybrid_dataset), 6)
            self.assertFalse(hasattr(materializer, "_anomaly_dataset"))
            self.assertFalse(hasattr(materializer, "_synth_anomaly_dataset"))
            self.assertFalse(hasattr(materializer, "_hybrid_dataset"))
            self.assertFalse(hasattr(materializer, "_num_anomaly_classes"))

            counts = materializer.repository.counts()
            self.assertEqual(
                counts,
                {
                    "original_samples": 4,
                    "real_anomalies": 4,
                    "synthetic_anomalies": 12,
                    "hybrid_samples": 6,
                    "placements": 12,
                },
            )
            hierarchy = materializer.repository.hierarchy()
            self.assertEqual(len(hierarchy), 12)
            for entry in hierarchy:
                self.assertEqual(
                    entry.synthetic_anomaly.real_anomaly_id,
                    entry.real_anomaly.id,
                )

            results = evaluate_study(config)
            self.assertEqual(results["glcm_cutout"]["sample_counter"], 12)
            self.assertEqual(results["volume_cutout"]["sample_counter"], 12)
            self.assertEqual(results["glcm_roi"]["sample_counter"], 12)
            with open(config.study.paths.metric_diffs_csv, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 36)
            self.assertTrue(all(row["real_anomaly_id"] for row in rows))
            self.assertTrue(all(row["synthetic_anomaly_id"] for row in rows))

    def test_3d_placements_use_explicit_zyx_columns(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration(
                "three-dimensional-study",
                "VAE_ResNet_3D",
                (1, 4, 4, 4),
                study_folder=str(Path(root) / "study"),
            )
            config.extraction.min_coverage_ratio = 0.0
            config.extraction.add_background_noise = False
            config.extraction.normalization = None
            config.extraction.roi.fixed_size = (4, 4, 4)
            config.generation.variants_per_real_anomaly = 2
            config.matching.routine = "fixed_from_extraction_control_fusion"
            config.matching.hybrids_per_original = 2
            config.matching.anomalies_per_hybrid = 1
            config.matching.reuse_synthetic_across_hybrids = False

            image = np.zeros((1, 12, 16, 16), dtype=np.float32)
            mask = np.zeros_like(image, dtype=np.uint8)
            image[:, 3:6, 5:8, 9:12] = 1.0
            mask[:, 3:6, 5:8, 9:12] = 1
            HybridDataGenerator(config).extract_anomalies(
                [InputSample(image, mask, "volume")]
            )
            HybridDataGenerator(
                config,
                generator_model=_FakeGenerator(),
            ).generate_synthetic_anomalies()
            planner = HybridDataGenerator(config)
            planner.plan_hybrid_samples(
                [
                    InputSample(
                        np.zeros_like(image),
                        np.zeros_like(mask),
                        "control-volume",
                    )
                ]
            )

            placements = planner.repository.list_placements()
            self.assertEqual(len(placements), 2)
            self.assertEqual(
                len({item.synthetic_anomaly_id for item in placements}), 2
            )
            for placement in placements:
                self.assertEqual(placement.spatial_dimensions, 3)
                self.assertIsNotNone(placement.position_z)
                self.assertEqual(len(placement.position), 3)

    def test_sibling_variants_can_share_a_hybrid_with_classical_fusion(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration(
                "sibling-variant-study",
                "VAE_ResNet_2D",
                (3, 8, 8),
                study_folder=str(Path(root) / "study"),
            )
            config.extraction.min_coverage_ratio = 0.0
            config.extraction.add_background_noise = False
            config.extraction.normalization = None
            config.extraction.roi.fixed_size = (8, 8)
            config.generation.variants_per_real_anomaly = 2
            config.matching.routine = "fixed_from_extraction_control_fusion"
            config.matching.hybrids_per_original = 1
            config.matching.anomalies_per_hybrid = 2
            config.matching.allow_sibling_variants_in_same_hybrid = True

            image = np.zeros((3, 24, 24), dtype=np.float32)
            mask = np.zeros((1, 24, 24), dtype=np.uint8)
            image[:, 8:12, 10:14] = 0.8
            mask[:, 8:12, 10:14] = 1

            HybridDataGenerator(config).extract_anomalies(
                [InputSample(image, mask, "anomaly")]
            )
            synthetic = HybridDataGenerator(
                config,
                generator_model=_FakeGenerator(),
            ).generate_synthetic_anomalies()
            planner = HybridDataGenerator(config)
            planned = planner.plan_hybrid_samples(
                [
                    InputSample(
                        image.copy(),
                        np.zeros_like(mask),
                        "anomaly",
                    )
                ]
            )

            self.assertEqual(len(synthetic), 2)
            self.assertEqual(len(planned), 1)
            self.assertEqual(planner.repository.counts()["original_samples"], 1)
            original = planner.repository.get_original_sample(
                planned[0].original_sample_id
            )
            self.assertGreater(
                planner.artifact_store.load_array(original.segmentation_path).sum(),
                0,
            )
            placements = planner.repository.list_placements(planned[0].id)
            self.assertEqual(len(placements), 2)
            self.assertEqual(
                {item.synthetic_anomaly_id for item in placements},
                {item.id for item in synthetic},
            )

            materializer = HybridDataGenerator(config)
            generated = materializer.materialize_hybrid_samples()
            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0].status, "generated")
            self.assertTrue(materializer.artifact_store.exists(generated[0].image_path))


if __name__ == "__main__":
    unittest.main()
