"""Postprocessing with the object-overlap rule on a study produced by the pipeline.

The images are in 0..255 units: a dark object on a bright background. In the
anomalous originals the object fills the image; in the controls it fills only
the left half. Variants are placed at the position of their real anomaly, so
anomalies from the right half of an original land on the background of a
control and must be rejected.
"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.domain.input_sample import InputSample
from hybrid_sample_generator.pipeline.hybrid_data_generator import HybridDataGenerator
from hybrid_sample_generator.postprocessing.service import decision_for
from tests.pipeline.test_quality_end_to_end import _IdentityGenerator

SIZE = 64
ON_OBJECT = ("left-0", "left-1")       # defects at x = 18: on the object of a control
OFF_OBJECT = ("right-0", "right-1")    # defects at x = 46: on the background of a control
CONTROLS = ("control-0", "control-1")


@contextlib.contextmanager
def _silent():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield


def _samples():
    rng = np.random.default_rng(9)
    samples = []
    for index, name in enumerate((*ON_OBJECT, *OFF_OBJECT, *CONTROLS)):
        image = np.full((1, SIZE, SIZE), 250.0, dtype=np.float32)
        mask = np.zeros((1, SIZE, SIZE), dtype=np.uint8)
        object_width = 32 if name in CONTROLS else SIZE
        image[0, :, :object_width] = 110.0 + 8.0 * rng.standard_normal((SIZE, object_width))
        if name not in CONTROLS:
            row, column = 20 + 6 * (index % 2), 18 if name in ON_OBJECT else 46
            image[0, row:row + 8, column:column + 8] -= 60.0
            mask[0, row:row + 8, column:column + 8] = 1
        samples.append(InputSample(image, mask, name))
    return samples


def _configuration(root, *, exempt=()):
    config = Configuration("overlap-study", study_folder=str(Path(root) / "study"))
    config.extraction.anomaly_size = (1, 16, 16)
    config.extraction.add_background_noise = False
    config.generation.variants_per_real_anomaly = 1
    # Every control receives every real anomaly once, at its source position.
    config.matching.routine = "fixed_from_extraction_control_fusion"
    config.matching.hybrids_per_original = 4
    post = config.postprocessing
    post.enabled = True
    post.action = "exclude"
    post.save_previews = True
    post.foreground.comparison = "below"
    post.foreground.threshold = 200
    post.rules.minimum_object_overlap.enabled = True
    post.rules.minimum_object_overlap.threshold = 0.9
    post.rules.minimum_object_overlap.exempt_source_names = list(exempt)
    return config


def _run(config):
    with _silent():
        generator = HybridDataGenerator(config, generator_model=_IdentityGenerator())
        generator.ingest_dataset(_samples())
        generator.extract_anomalies()
        generator.generate_synthetic_anomalies()
        generator.plan_hybrid_samples()
        generator.materialize_hybrid_samples()
        summary = generator.postprocess_hybrid_samples()
    return generator, summary


def _donor(generator, hybrid):
    repository = generator.repository
    placement = repository.list_placements(hybrid.id)[0]
    synthetic = repository.get_synthetic_anomaly(placement.synthetic_anomaly_id)
    real = repository.get_real_anomaly(synthetic.real_anomaly_id)
    return repository.get_original_sample(real.original_sample_id).source_name


class PostprocessingEndToEndTests(unittest.TestCase):
    def _decisions(self, generator, config):
        repository, store = generator.repository, generator.artifact_store
        result = {}
        for hybrid in repository.list_hybrid_samples():
            original = repository.get_original_sample(hybrid.original_sample_id)
            decision = decision_for(repository, store, hybrid, original, config.postprocessing)
            result[hybrid.id] = (_donor(generator, hybrid), decision)
        return result

    def test_anomalies_on_the_background_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root)
            generator, summary = _run(config)

            self.assertEqual(summary["total"], 8)
            self.assertEqual(summary["passed"], 4)
            self.assertEqual(summary["failed"], 4)
            self.assertEqual(summary["excluded"], 4)

            decisions = self._decisions(generator, config)
            self.assertEqual(len(decisions), 8)
            for donor, decision in decisions.values():
                reasons = json.loads(decision["reasons_json"])
                measurements = json.loads(decision["measurements_json"])
                if donor in ON_OBJECT:
                    self.assertTrue(decision["accepted"])
                    self.assertEqual(reasons, [])
                    self.assertGreaterEqual(measurements["object_overlap"], 0.9)
                else:
                    self.assertFalse(decision["accepted"])
                    self.assertEqual(reasons, ["minimum_object_overlap"])
                    self.assertLess(measurements["object_overlap"], 0.1)

            # Rejected hybrids stay in the study; only the decision marks them.
            hybrids = generator.repository.list_hybrid_samples(status="generated")
            self.assertEqual(len(hybrids), 8)
            self.assertEqual(len(generator.datasets.hybrid_samples()), 8)

            report = Path(summary["report_folder"])
            with (report / "decisions.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 8)
            self.assertEqual(sum(row["accepted"] == "True" for row in rows), 4)
            self.assertEqual(len(list((report / "previews").glob("*.png"))), 8)
            self.assertEqual(json.loads((report / "summary.json").read_text())["failed"], 4)

    def test_exempt_source_originals_pass_the_overlap_rule(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root, exempt=OFF_OBJECT)
            generator, summary = _run(config)

            self.assertEqual(summary["passed"], 8)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["unknown_exempt_source_names"], [])
            for donor, decision in self._decisions(generator, config).values():
                self.assertTrue(decision["accepted"])
                measurements = json.loads(decision["measurements_json"])
                self.assertEqual(measurements["exempt_placements"], 1 if donor in OFF_OBJECT else 0)

    def test_exemptions_added_after_fusion_require_new_materialization(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root)
            generator, _ = _run(config)
            config.postprocessing.rules.minimum_object_overlap.exempt_source_names = list(OFF_OBJECT)

            with _silent(), self.assertRaisesRegex(ValueError, "Rematerialize"):
                HybridDataGenerator(config).postprocess_hybrid_samples()

            # The earlier decisions no longer match the configuration.
            hybrid = generator.repository.list_hybrid_samples()[0]
            original = generator.repository.get_original_sample(hybrid.original_sample_id)
            with self.assertRaisesRegex(ValueError, "stale"):
                decision_for(
                    generator.repository, generator.artifact_store, hybrid, original,
                    config.postprocessing,
                )

            with _silent():
                fresh = HybridDataGenerator(config, generator_model=_IdentityGenerator())
                fresh.materialize_hybrid_samples()
                summary = fresh.postprocess_hybrid_samples()
            self.assertEqual(summary["passed"], 8)

    def test_unknown_exempt_names_are_reported(self):
        with tempfile.TemporaryDirectory() as root, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = _configuration(root, exempt=("missing.png",))
            _, summary = _run(config)

        self.assertEqual(summary["unknown_exempt_source_names"], ["missing.png"])
        self.assertEqual(summary["failed"], 4)
        self.assertTrue(any("match no ingested original" in str(item.message) for item in caught))

    def test_rematerialization_clears_decisions_and_report_mode_excludes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            config = _configuration(root)
            generator, first = _run(config)
            with _silent():
                generator.materialize_hybrid_samples()
            hybrid = generator.repository.list_hybrid_samples()[0]
            original = generator.repository.get_original_sample(hybrid.original_sample_id)
            with self.assertRaisesRegex(ValueError, "missing or stale"):
                decision_for(
                    generator.repository, generator.artifact_store, hybrid, original,
                    config.postprocessing,
                )

            config.postprocessing.action = "report"
            with _silent():
                second = HybridDataGenerator(config).postprocess_hybrid_samples()
            self.assertEqual(second["failed"], 4)
            self.assertEqual(second["excluded"], 0)
            # Each run keeps its own report folder.
            self.assertNotEqual(first["report_folder"], second["report_folder"])
            self.assertTrue(Path(first["report_folder"]).is_dir())

            config.postprocessing.enabled = False
            with _silent():
                self.assertEqual(HybridDataGenerator(config).postprocess_hybrid_samples(), {"enabled": False})


if __name__ == "__main__":
    unittest.main()
