"""Opening/viewing old MVTec studies must not prepare splits or rewrite files."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.mvtec_ad2.presets import create_configuration
from examples.mvtec_ad2.studies import find_studies
from examples.mvtec_ad2.studies import open_study
from examples.mvtec_ad2.review import review_studies
from hybrid_sample_generator.persistence.study_repository import StudyRepository


class LegacyViewerTests(unittest.TestCase):
    def test_read_paths_accept_no_manifest_or_source_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = create_configuration("can", save_path=root / "can")
            path = Path(config.save_config_file())
            StudyRepository(config.study.paths.artifact_database)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with patch("examples.mvtec_ad2.studies.prepare_studies", side_effect=AssertionError("No training preparation")), patch("hybrid_sample_generator.visualization.run_hybrid_visualizer") as viewer, patch("hybrid_sample_generator.evaluation.service.evaluate_study") as evaluate, patch("examples.mvtec_ad2.studies.create_configuration", side_effect=AssertionError("No presets")):
                review_studies([path.parent])
                review_studies([path.parent], actions=("evaluate_generator", "visualize"))
                self.assertEqual(viewer.call_count, 2)
                self.assertEqual(evaluate.call_count, 1)
                folders = find_studies(root)
            self.assertEqual(folders, [path.parent])
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            case = open_study(path.parent)
            self.assertIsNone(case.sample_dataloader)
            self.assertIsNone(case.split_manifest)

    def test_missing_study_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "missing"
            for operation in (open_study, find_studies):
                with self.assertRaises(FileNotFoundError):
                    operation(output)
            self.assertFalse(output.exists())
