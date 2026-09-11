"""End-to-end tests for generation and evaluation workflows."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.evaluation.service import evaluate_study
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from tests.pipeline.support import (
    _FakeFusionBackend,
    _FakeGenerator,
    _anomaly_samples,
    _control_samples,
)


class PipelineEndToEndTests(unittest.TestCase):
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

            ingestor = HybridDataGenerator(config)
            summary = ingestor.ingest_dataset(
                [*_anomaly_samples(), *_control_samples()]
            )
            self.assertEqual(summary.total_samples, 4)
            self.assertEqual(summary.anomalous_samples, 2)
            self.assertEqual(summary.control_samples, 2)

            extractor = HybridDataGenerator(config)
            real = extractor.extract_anomalies()
            self.assertEqual(len(real), 4)

            generator = HybridDataGenerator(config, generator_model=_FakeGenerator())
            synthetic = generator.generate_synthetic_anomalies()
            self.assertEqual(len(synthetic), 12)
            self.assertEqual(
                {item.variant_index for item in synthetic}, {0, 1, 2}
            )

            planner = HybridDataGenerator(config)
            planned = planner.plan_hybrid_samples()
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
            with patch(
                "hybrid_sample_generator.pipeline.hybrid_data_generator.tqdm",
                side_effect=lambda iterable, **_kwargs: iterable,
            ) as progress:
                generated = materializer.materialize_hybrid_samples()
            progress.assert_called_once()
            progress_args, progress_kwargs = progress.call_args
            self.assertEqual(len(progress_args[0]), 6)
            self.assertEqual(progress_kwargs["desc"], "Materializing hybrid samples")
            self.assertEqual(progress_kwargs["unit"], "sample")
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
