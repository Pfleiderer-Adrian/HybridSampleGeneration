"""Input failures, stage invalidation, and complete 3D pipeline tests."""
import csv
import sqlite3
from dataclasses import fields
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from hybrid_sample_generator.evaluation.service import evaluate_study
from tests.pipeline.support import _FakeGenerator, _anomaly_samples, _control_samples


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = Configuration('regression', study_folder=self.temporary.name)
        self.config.extraction.anomaly_size = (1, 8, 8)
        self.config.extraction.roi.fixed_size = (8, 8)
        self.config.extraction.min_coverage_ratio = 0
        self.config.extraction.normalization = None
        self.config.extraction.add_background_noise = False
        self.config.generation.variants_per_real_anomaly = 1
        self.config.matching.routine = 'fixed_from_extraction_control_fusion'
        self.samples = [*_anomaly_samples(), *_control_samples()]
        self.pipeline = HybridDataGenerator(self.config, generator_model=_FakeGenerator())

    def prepare(self):
        self.pipeline.ingest_dataset(self.samples)
        self.pipeline.extract_anomalies()
        self.pipeline.generate_synthetic_anomalies()
        self.pipeline.plan_hybrid_samples()
        self.pipeline.materialize_hybrid_samples()

    def test_invalid_inputs_preserve_existing_catalog(self):
        self.pipeline.ingest_dataset(self.samples)
        before = self.pipeline.repository.list_original_samples()
        image = np.ones((1, 8, 8))
        cases = [[], [InputSample(image, None, 'same')] * 2,
                 [InputSample(image, None, 'a', source_image_path='/tmp/shared'), InputSample(image, None, 'b', source_image_path='/tmp/shared')],
                 [InputSample(np.ones((2, 8, 8)), None, 'channels')],
                 [InputSample(image, np.ones((1, 7, 8)), 'mask')],
                 [InputSample(np.ones((8, 8)), None, 'dimensions')]]
        for samples in cases:
            with self.subTest(samples=[s.source_name for s in samples]), self.assertRaises(ValueError):
                self.pipeline.ingest_dataset(samples)
            self.assertEqual(self.pipeline.repository.list_original_samples(), before)

    def test_ingestion_write_failure_preserves_existing_catalog_and_artifacts(self):
        self.prepare()
        before = self.pipeline.repository.list_original_samples()
        arrays = {record.id: self.pipeline.artifact_store.load_array(record.image_path).copy() for record in before}
        counts = self.pipeline.repository.counts()
        with patch.object(self.pipeline.artifact_store, "save_entity_array", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.pipeline.ingest_dataset([InputSample(np.ones((1, 8, 8)), None, "new-source")])
        self.assertEqual(self.pipeline.repository.list_original_samples(), before)
        self.assertEqual(self.pipeline.repository.counts(), counts)
        for record in before:
            np.testing.assert_array_equal(self.pipeline.artifact_store.load_array(record.image_path), arrays[record.id])

    def test_each_repeated_stage_invalidates_only_its_downstream_records(self):
        for stage in ('ingest', 'extract', 'generate'):
            with self.subTest(stage=stage):
                self.prepare()
                if stage == 'ingest':
                    self.pipeline.ingest_dataset(self.samples)
                elif stage == 'extract':
                    self.pipeline.extract_anomalies()
                else:
                    self.pipeline.generate_synthetic_anomalies()
                counts = self.pipeline.repository.counts()
                self.assertEqual(counts['original_samples'], 4)
                self.assertEqual(counts['real_anomalies'], 0 if stage == 'ingest' else 4)
                self.assertEqual(counts['synthetic_anomalies'], 4 if stage == 'generate' else 0)
                self.assertEqual(counts['hybrid_samples'], 0)
                self.assertEqual(counts['placements'], 0)

    def test_extraction_and_generation_failures_remove_stale_downstream_and_can_retry(self):
        for stage in ('extract', 'generate'):
            with self.subTest(stage=stage):
                self.prepare()
                target = ('hybrid_sample_generator.extraction.service.crop_and_center_anomalies' if stage == 'extract' else 'tests.pipeline.support._FakeGenerator.generate')
                method = (self.pipeline.extract_anomalies if stage == 'extract' else self.pipeline.generate_synthetic_anomalies)
                with patch(target, side_effect=RuntimeError('stage failure')), self.assertRaises(RuntimeError):
                    method()
                counts = self.pipeline.repository.counts()
                self.assertEqual(counts['hybrid_samples'], 0)
                self.assertEqual(counts['placements'], 0)
                self.assertEqual(counts['synthetic_anomalies'], 0)
                self.assertEqual(counts['real_anomalies'], 0 if stage == 'extract' else 4)
                self.assertTrue(method())

    def test_partial_generation_failure_keeps_valid_records_and_retry_is_repeatable(self):
        self.prepare()
        original = self.pipeline.repository.list_synthetic_anomalies()
        expected = {record.id: self.pipeline.artifact_store.load_array(record.image_path).copy() for record in original}
        generate = _FakeGenerator.generate
        calls = 0

        def fail_second(model, sample, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("interrupted generation")
            return generate(model, sample, **kwargs)

        with patch.object(_FakeGenerator, "generate", fail_second), self.assertRaises(RuntimeError):
            self.pipeline.generate_synthetic_anomalies()
        partial = self.pipeline.repository.list_synthetic_anomalies()
        self.assertEqual(len(partial), 1)
        self.assertEqual(self.pipeline.repository.counts()["placements"], 0)
        np.testing.assert_array_equal(self.pipeline.artifact_store.load_array(partial[0].image_path), expected[partial[0].id])
        restarted = HybridDataGenerator(self.config, generator_model=_FakeGenerator())
        repeated = restarted.generate_synthetic_anomalies()
        self.assertEqual({record.id: record for record in repeated}, {record.id: record for record in original})
        for record in repeated:
            np.testing.assert_array_equal(restarted.artifact_store.load_array(record.image_path), expected[record.id])

    def test_partial_extraction_failure_keeps_only_complete_records_and_can_retry(self):
        from hybrid_sample_generator.extraction.service import crop_and_center_anomalies
        self.prepare()
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("interrupted extraction")
            return crop_and_center_anomalies(*args, **kwargs)

        with patch("hybrid_sample_generator.extraction.service.crop_and_center_anomalies", side_effect=fail_second), self.assertRaises(RuntimeError):
            self.pipeline.extract_anomalies()
        partial = self.pipeline.repository.list_real_anomalies()
        self.assertEqual(len(partial), 2)
        for record in partial:
            for path in (record.image_path, record.segmentation_path, record.roi_image_path, record.roi_segmentation_path):
                self.assertTrue(self.pipeline.artifact_store.exists(path))
        self.assertEqual(self.pipeline.repository.counts()["synthetic_anomalies"], 0)
        self.assertEqual(len(self.pipeline.extract_anomalies()), 4)

    def test_full_3d_pipeline_with_classical_fusion_and_repeatable_evaluation(self):
        self.config.model.set_model('VAE_ResNet_3D')
        self.config.extraction.anomaly_size = (1, 4, 4, 4)
        self.config.extraction.roi.fixed_size = (4, 4, 4)
        image = np.full((1, 12, 12, 12), .2, dtype=np.float32)
        mask = np.zeros_like(image, dtype=np.uint8)
        image[:, 4:7, 4:7, 4:7] = .8
        mask[:, 4:7, 4:7, 4:7] = 1
        pipeline = HybridDataGenerator(self.config, generator_model=_FakeGenerator())
        pipeline.ingest_dataset([InputSample(image, mask, 'anomaly'), InputSample(np.full_like(image, .2), np.zeros_like(mask), 'control')])
        pipeline.extract_anomalies()
        pipeline.generate_synthetic_anomalies()
        pipeline.plan_hybrid_samples()
        generated = pipeline.materialize_hybrid_samples()
        self.assertEqual(len(generated), 1)
        hybrid = generated[0]
        self.assertEqual(hybrid.status, 'generated')
        self.assertEqual(pipeline.artifact_store.load_array(hybrid.image_path).shape, image.shape)
        self.assertGreater(pipeline.artifact_store.load_array(hybrid.segmentation_path).sum(), 0)
        placement = pipeline.repository.list_placements()[0]
        self.assertIsNotNone(placement.position_z)
        results = evaluate_study(self.config)
        self.assertEqual([result['sample_counter'] for result in results.values()], [1, 1, 1])
        path = self.config.study.paths.metric_diffs_csv
        first = Path(path).read_text()
        evaluate_study(self.config)
        self.assertEqual(Path(path).read_text(), first)
        with open(path, newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def assert_all_record_artifact_references_are_complete(self):
        repository = self.pipeline.repository
        for records in (repository.list_original_samples(), repository.list_real_anomalies(), repository.list_synthetic_anomalies(), repository.list_hybrid_samples(), repository.list_placements()):
            for record in records:
                for field in fields(record):
                    if field.name.endswith("_path"):
                        path = getattr(record, field.name)
                        if path is not None:
                            self.assertTrue(self.pipeline.artifact_store.exists(path), (record.id, field.name))
                for image_role, mask_role in (("image_path", "segmentation_path"), ("roi_image_path", "roi_segmentation_path")):
                    image_path = getattr(record, image_role, None)
                    mask_path = getattr(record, mask_role, None)
                    if image_path is not None and mask_path is not None:
                        image = self.pipeline.artifact_store.load_array(image_path)
                        mask = self.pipeline.artifact_store.load_array(mask_path)
                        self.assertEqual(image.shape[1:], mask.shape[1:])

    def test_late_artifact_failures_do_not_leave_incomplete_record_references(self):
        for stage in ("extract", "generate", "fusion"):
            with self.subTest(stage=stage):
                self.prepare()
                save = self.pipeline.artifact_store.save_entity_array
                calls = 0

                def fail_fourth(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 4:
                        raise OSError("late artifact write failure")
                    return save(*args, **kwargs)

                method = {"extract": self.pipeline.extract_anomalies, "generate": self.pipeline.generate_synthetic_anomalies, "fusion": self.pipeline.materialize_hybrid_samples}[stage]
                with patch.object(self.pipeline.artifact_store, "save_entity_array", side_effect=fail_fourth), self.assertRaises(RuntimeError if stage == "fusion" else OSError):
                    method()
                self.assert_all_record_artifact_references_are_complete()
                self.assertTrue(method())
                self.assert_all_record_artifact_references_are_complete()

    def test_database_record_failures_leave_valid_references_and_can_retry(self):
        for stage, table in (("extract", "real_anomalies"), ("generate", "synthetic_anomalies"), ("fusion", "placements")):
            with self.subTest(stage=stage):
                self.prepare()
                with self.pipeline.repository.connection() as connection:
                    connection.execute(f"CREATE TRIGGER reject_record BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT, 'injected record failure'); END")
                method = {"extract": self.pipeline.extract_anomalies, "generate": self.pipeline.generate_synthetic_anomalies, "fusion": self.pipeline.materialize_hybrid_samples}[stage]
                try:
                    with self.assertRaises(RuntimeError if stage == "fusion" else sqlite3.IntegrityError):
                        method()
                    self.assert_all_record_artifact_references_are_complete()
                finally:
                    with self.pipeline.repository.connection() as connection:
                        connection.execute("DROP TRIGGER reject_record")
                self.assertTrue(method())
                self.assert_all_record_artifact_references_are_complete()
