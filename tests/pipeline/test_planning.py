import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from skimage.feature import match_template as skimage_match_template

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from tests.pipeline.support import _FakeGenerator


class PipelinePlanningTests(unittest.TestCase):
    def test_local_matching_reuses_cached_full_image_results(self):
        with tempfile.TemporaryDirectory() as root:
            config = Configuration(
                "local-cache-study",
                "VAE_ResNet_2D",
                (1, 8, 8),
                study_folder=str(Path(root) / "study"),
            )
            config.extraction.min_coverage_ratio = 0.0
            config.extraction.add_background_noise = False
            config.extraction.normalization = None
            config.extraction.roi.fixed_size = (8, 8)
            config.generation.variants_per_real_anomaly = 1
            config.matching.routine = "local"
            config.matching.hybrids_per_original = 1
            config.matching.anomalies_per_hybrid = 1

            anomaly_image = np.zeros((1, 32, 32), dtype=np.float32)
            anomaly_mask = np.zeros_like(anomaly_image, dtype=np.uint8)
            anomaly_image[:, 4:8, 4:8] = 1.0
            anomaly_mask[:, 4:8, 4:8] = 1
            control_image = np.zeros_like(anomaly_image)
            control_image[:, 23:27, 23:27] = 1.0

            HybridDataGenerator(config).ingest_dataset(
                [
                    InputSample(anomaly_image, anomaly_mask, "anomaly"),
                    InputSample(
                        control_image,
                        np.zeros_like(anomaly_mask),
                        "control",
                    ),
                ]
            )
            HybridDataGenerator(config).extract_anomalies()
            HybridDataGenerator(
                config,
                generator_model=_FakeGenerator(),
            ).generate_synthetic_anomalies()

            planner = HybridDataGenerator(config)
            with patch(
                "hybrid_sample_generator.matching.template_matching.match_template",
                wraps=skimage_match_template,
            ) as matching_call:
                first_plan = planner.plan_hybrid_samples()
            self.assertGreater(matching_call.call_count, 0)
            self.assertEqual(planner.repository.count_match_candidates(), 1)

            first_placement = planner.repository.list_placements(first_plan[0].id)[0]
            self.assertGreater(first_placement.position_y, 0.5)
            self.assertGreater(first_placement.position_x, 0.5)

            with patch(
                "hybrid_sample_generator.matching.template_matching.match_template",
                side_effect=AssertionError("cached matches must not be recomputed"),
            ):
                second_plan = HybridDataGenerator(config).plan_hybrid_samples()
            second_placement = planner.repository.list_placements(second_plan[0].id)[0]
            self.assertEqual(first_placement.position, second_placement.position)

            for routine in ("global", "batchwise"):
                config.matching.routine = routine
                with patch(
                    "hybrid_sample_generator.matching.template_matching.match_template",
                    side_effect=AssertionError("compatible routines must reuse pair matches"),
                ):
                    cached_plan = HybridDataGenerator(config).plan_hybrid_samples()
                cached_placement = planner.repository.list_placements(cached_plan[0].id)[0]
                self.assertEqual(cached_placement.method, routine)
                self.assertEqual(first_placement.position, cached_placement.position)

            HybridDataGenerator(config).extract_anomalies()
            self.assertEqual(planner.repository.count_match_candidates(), 0)
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
            HybridDataGenerator(config).ingest_dataset(
                [
                    InputSample(image, mask, "volume"),
                    InputSample(
                        np.zeros_like(image),
                        np.zeros_like(mask),
                        "control-volume",
                    ),
                ]
            )
            HybridDataGenerator(config).extract_anomalies()
            HybridDataGenerator(
                config,
                generator_model=_FakeGenerator(),
            ).generate_synthetic_anomalies()
            planner = HybridDataGenerator(config)
            planner.plan_hybrid_samples()

            placements = planner.repository.list_placements()
            self.assertEqual(len(placements), 2)
            self.assertEqual(
                len({item.synthetic_anomaly_id for item in placements}), 2
            )
            for placement in placements:
                self.assertEqual(placement.spatial_dimensions, 3)
                self.assertIsNotNone(placement.position_z)
                self.assertEqual(len(placement.position), 3)
