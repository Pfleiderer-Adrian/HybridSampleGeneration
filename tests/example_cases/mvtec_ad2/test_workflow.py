"""Regression coverage for explicit steps, study access and optional export."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from examples.mvtec_ad2.presets import create_configuration
from examples.mvtec_ad2.dataloader import MVTecAD2Dataloader
from examples.mvtec_ad2.export import segmentation_for_png
from examples.mvtec_ad2.studies import open_study, prepare_studies
from examples.mvtec_ad2.pipeline import run_new_experiment, run_existing_studies, run_study
from examples.mvtec_ad2.configuration import Experiment, SplitConfiguration
from examples.mvtec_ad2.splits import manifest_samples, verify_repository
from examples.mvtec_ad2.steps import DEFAULT_GENERATION_STEPS
from .test_downstream import create_images, seed_hybrid
from examples.mvtec_ad2.workflow import preflight


class WorkflowTests(unittest.TestCase):
    def test_new_experiment_persists_selected_split_and_category_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            experiment = Experiment(
                root, root / "studies", ("can",),
                SplitConfiguration(test_enabled=False, seed=17),
            )
            results = run_new_experiment(experiment, steps=("ingest",))
            self.assertEqual(len(results), 1)
            saved = open_study(results[0].study_folder)
            self.assertFalse(saved.split_manifest["split"]["test_enabled"])
            self.assertEqual(saved.config.study.seed, 17)
            self.assertEqual(saved.config.generation.variation_strength, 1.5)
            self.assertEqual(saved.config.downstream.training.batch_size, 8)

    def test_invalid_selection_fails_before_study_preparation(self):
        with patch("examples.mvtec_ad2.pipeline.prepare_studies") as prepare:
            with self.assertRaises(ValueError):
                run_new_experiment(Experiment(Path("unused"), Path("unused"), ("can",)), steps=("train",))
            with self.assertRaises(ValueError):
                run_new_experiment(Experiment(Path("unused"), Path("unused"), ("can",)), steps=("evaluate_downstream",))
            with self.assertRaises(ValueError):
                run_existing_studies(study_folders=[], steps=("export",))
            prepare.assert_not_called()

    def test_generation_has_optional_export_and_saves_configuration_once(self):
        for export in (False, True):
            with self.subTest(export=export), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                create_images(root)
                case = prepare_studies(root, "can", save_path=root / "studies")[0]
                steps = (*DEFAULT_GENERATION_STEPS, "export") if export else DEFAULT_GENERATION_STEPS
                with patch("examples.mvtec_ad2.pipeline.HybridDataGenerator") as generator, patch("examples.mvtec_ad2.export.export_hybrids") as save_images, patch.object(case.config, "save_config_file", wraps=case.config.save_config_file) as save_config:
                    result = run_study(case, steps=steps)
                    generator.return_value.materialize_hybrid_samples.assert_called_once_with()
                    generator.return_value.train_generator.assert_called_once_with()
                    generator.return_value.load_generator.assert_not_called()
                    self.assertEqual(save_images.call_count, int(export))
                    save_config.assert_called_once_with()
                self.assertIsNone(result.downstream_run_folder)
                self.assertFalse((result.study_folder / "exports").exists())

    def test_export_existing_artifacts_without_generator_or_config_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            case = prepare_studies(root, "can", save_path=root / "studies")[0]
            seed_hybrid(case)
            config_path = Path(case.config.save_config_file())
            before = config_path.read_bytes()
            # Legacy exports, like viewing, do not require a split manifest.
            legacy = replace(case, split_manifest=None)
            with patch("examples.mvtec_ad2.pipeline.HybridDataGenerator", side_effect=AssertionError("No generator")):
                run_study(legacy, steps=("export",))
            self.assertEqual(config_path.read_bytes(), before)
            images = list(Path(case.config.study.paths.generated_images).glob("*.png"))
            masks = list(Path(case.config.study.paths.generated_segmentations).glob("*.png"))
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].name, masks[0].name)
            self.assertEqual(set(np.unique(np.asarray(Image.open(masks[0])))), {0, 255})

    def test_dataloader_iterators_share_identical_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            case = prepare_studies(root, "can", save_path=root / "studies")[0]
            loader = MVTecAD2Dataloader(manifest_samples(case.split_manifest, "train"))
            for (image, mask, name), record in zip(loader, loader.iter_input_samples()):
                np.testing.assert_array_equal(image, record.image)
                np.testing.assert_array_equal(mask, record.segmentation)
                self.assertEqual(name, record.source_name)
            np.testing.assert_array_equal(segmentation_for_png(np.array([[[0., .5]]])), [[[0, 255]]])

    def test_preflight_reuses_hybrid_pairs_and_rejects_heldout_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            case = prepare_studies(root, "can", save_path=root / "studies")[0]
            case.config.downstream.data.hybrid_fraction = 1
            repo, _ = seed_hybrid(case)
            with patch("examples.mvtec_ad2.downstream.datasets.verify_repository", wraps=verify_repository) as verify:
                pairs = preflight(case, ("train_downstream",))
                self.assertEqual(len(pairs), 1)
                verify.assert_called_once()
            bad = repo.list_original_samples()[0]
            bad.metadata["source_image_path"] = case.split_manifest["partitions"]["validation"][0]["image_path"]
            repo.replace_original_samples([bad])
            with self.assertRaisesRegex(ValueError, "held-out"):
                preflight(case, ("train_downstream",))

    def test_relocated_study_uses_actual_folder_and_saved_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = create_configuration("can", save_path=root / "original")
            path = Path(config.save_config_file())
            moved = root / "arbitrary-name"
            path.parent.rename(moved)
            with patch("examples.mvtec_ad2.studies.create_configuration", side_effect=AssertionError("No presets")):
                loaded = open_study(moved)
            self.assertEqual(Path(loaded.config.study.folder), moved)
            self.assertEqual(loaded.category, "can")

    def test_standalone_evaluation_does_not_open_generator_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = create_configuration("can", save_path=temporary)
            from examples.mvtec_ad2.records import MVTecAD2Study
            case = MVTecAD2Study("can", None, config, None, {"fingerprint": "test"})
            with patch("examples.mvtec_ad2.downstream.runner.load_run") as load, patch("examples.mvtec_ad2.workflow.StudyRepository", side_effect=AssertionError("No repository")):
                preflight(case, ("evaluate_downstream",), downstream_run_id="run")
                load.assert_called_once_with(case, "run")

    def test_preflight_accounts_for_invalidated_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            case = prepare_studies(root, "can", save_path=root / "studies")[0]
            seed_hybrid(case)
            for steps in (("ingest", "train_generator"), ("extract", "plan"),
                          ("ingest", "materialize"), ("plan", "export"),
                          ("ingest", "train_downstream")):
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    preflight(case, steps)

    def test_legacy_training_and_missing_prerequisites_fail_early(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_images(root)
            case = prepare_studies(root, "can", save_path=root / "studies")[0]
            for steps in (("extract",), ("train_generator",), ("plan",), ("materialize",), ("export",), ("load_generator",)):
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    preflight(case, steps)
            with self.assertRaisesRegex(ValueError, "split manifest"):
                preflight(replace(case, split_manifest=None), ("train_downstream",))
