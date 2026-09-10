import tempfile
import unittest
from pathlib import Path

import numpy as np

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from tests.pipeline.support import _FakeGenerator


class PipelineMaterializationTests(unittest.TestCase):
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
            config.matching.routine = "fixed_from_extraction_anomaly_fusion"
            config.matching.hybrids_per_original = 1
            config.matching.anomalies_per_hybrid = 2
            config.matching.allow_sibling_variants_in_same_hybrid = True

            image = np.zeros((3, 24, 24), dtype=np.float32)
            mask = np.zeros((1, 24, 24), dtype=np.uint8)
            image[:, 8:12, 10:14] = 0.8
            mask[:, 8:12, 10:14] = 1

            HybridDataGenerator(config).ingest_dataset(
                [InputSample(image, mask, "anomaly")]
            )
            HybridDataGenerator(config).extract_anomalies()
            synthetic = HybridDataGenerator(
                config,
                generator_model=_FakeGenerator(),
            ).generate_synthetic_anomalies()
            planner = HybridDataGenerator(config)
            planned = planner.plan_hybrid_samples()

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
