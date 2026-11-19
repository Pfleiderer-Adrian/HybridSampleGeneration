"""Tests for declared parameter metadata and the configuration schema."""

import json
import unittest
from dataclasses import fields, is_dataclass

from hybrid_sample_generator.configuration import Configuration, configuration_schema
from hybrid_sample_generator.configuration.parameters import (
    check_value,
    iter_parameters,
)
from hybrid_sample_generator.configuration.matching import MatchingConfiguration
from hybrid_sample_generator.configuration.training import TrainingConfiguration
from hybrid_sample_generator.fusion.registry import FUSION_BACKEND_REGISTRY
from hybrid_sample_generator.generation.model_settings import SearchSpace
from hybrid_sample_generator.generation.registry import MODEL_REGISTRY


def _leaf_names(instance):
    names = set()
    for item in fields(instance):
        value = getattr(instance, item.name)
        if is_dataclass(value):
            names.update(f"{item.name}.{name}" for name in _leaf_names(value))
        else:
            names.add(item.name)
    return names


class ConfigurationSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = configuration_schema()
        cls.sections = cls.schema["sections"]

    def test_schema_is_json_compatible_and_versioned(self):
        self.assertEqual(json.loads(json.dumps(self.schema)), self.schema)
        self.assertEqual(self.schema["schema_version"], Configuration.SCHEMA_VERSION)
        self.assertEqual(Configuration.schema(), self.schema)

    def test_every_configuration_field_is_described(self):
        config = Configuration("schema-test", study_folder="results/schema-test")
        expected = set(vars(config)) - {"schema_version"}
        self.assertEqual(set(self.sections), expected)
        for name in expected - {"model", "fusion"}:
            with self.subTest(section=name):
                described = {path for path, _ in iter_parameters(self.sections[name])}
                self.assertEqual(described, _leaf_names(getattr(config, name)))

    def test_every_parameter_has_a_type_and_description(self):
        groups = [(name, group) for name, group in self.sections.items()
                  if group["type"] == "group"]
        for section in ("model", "fusion"):
            for name, variant in self.sections[section]["variants"].items():
                groups.append((f"{section}[{name}]", variant["parameters"]))
        for prefix, group in groups:
            for path, entry in iter_parameters(group, prefix):
                with self.subTest(parameter=path):
                    self.assertIn("type", entry)
                    self.assertTrue(entry.get("doc"))
                    if not entry.get("required") and "default_label" not in entry:
                        self.assertIn("default", entry)

    def test_defaults_satisfy_their_own_descriptors(self):
        for section in ("model", "fusion"):
            for name, variant in self.sections[section]["variants"].items():
                for path, entry in iter_parameters(variant["parameters"], name):
                    if entry.get("validated", True):
                        check_value(entry["default"], entry, path)

    def test_ranges_choices_and_nullability_are_reported(self):
        matching = self.sections["matching"]["fields"]
        self.assertEqual(matching["batch_size"], {
            "type": "int", "min": 1, "default": 32,
            "doc": matching["batch_size"]["doc"],
        })
        self.assertEqual(matching["routine"]["type"], "choice")
        self.assertIn("local", matching["routine"]["choices"])

        training = self.sections["training"]["fields"]
        ratio = training["validation_ratio"]
        self.assertEqual((ratio["min"], ratio["max"], ratio["max_exclusive"]), (0.0, 1.0, True))
        self.assertTrue(training["gradient_clip_norm"]["nullable"])
        self.assertTrue(training["gradient_clip_norm"]["min_exclusive"])
        self.assertEqual(training["trial_selection"]["type"], "any_of")
        self.assertEqual(training["early_stopping"]["keys"]["patience"]["min"], 0)

        size = self.sections["extraction"]["fields"]["anomaly_size"]
        self.assertEqual((size["type"], size["min_length"], size["max_length"]), ("list", 3, 4))
        self.assertEqual(size["items"], {"type": "int", "min": 1})
        self.assertEqual(size["default"], [3, 64, 64])

        study = self.sections["study"]["fields"]
        self.assertTrue(study["name"]["required"])
        self.assertNotIn("default", study["name"])
        self.assertIn("default_label", study["folder"])

    def test_mask_transforms_are_marked_for_a_dedicated_editor(self):
        group = self.sections["augmentation"]["fields"]["mask_transforms"]
        self.assertEqual(group["type"], "group")
        self.assertTrue(group["custom"])

    def test_model_and_fusion_variants_follow_the_registries(self):
        model = self.sections["model"]
        self.assertEqual(set(model["variants"]), set(MODEL_REGISTRY))
        self.assertEqual(model["selector"], "name")
        self.assertIn(model["default"], model["variants"])
        for name, spec in MODEL_REGISTRY.items():
            with self.subTest(model=name):
                variant = model["variants"][name]
                parameters = spec.build_configuration()
                self.assertEqual(variant["spatial_dims"], spec.spatial_dims)
                self.assertEqual(
                    set(variant["parameters"]["fields"]),
                    {item.name for item in fields(parameters)},
                )
                self.assertEqual(
                    variant["search"]["default"],
                    json.loads(json.dumps(spec.build_search_space(parameters).to_dict())),
                )
                # Every advertised distribution type is accepted by SearchSpace.
                restored = SearchSpace.from_dict(parameters, variant["search"]["default"])
                for parameter in restored.names():
                    self.assertIn(
                        variant["search"]["default"][parameter]["type"],
                        variant["search"]["distributions"][parameter],
                    )

        fusion = self.sections["fusion"]
        self.assertEqual(set(fusion["variants"]), set(FUSION_BACKEND_REGISTRY))
        self.assertEqual(fusion["selector"], "backend")
        self.assertIn(
            "guidance_mode", fusion["variants"]["poisson"]["parameters"]["fields"]
        )

    def test_2d_and_3d_variants_report_their_own_defaults(self):
        variants = self.sections["model"]["variants"]
        two = variants["cVAE_ConvNeXt_2D"]["parameters"]["fields"]["z_channels"]["default"]
        three = variants["cVAE_ConvNeXt_3D"]["parameters"]["fields"]["z_channels"]["default"]
        self.assertEqual(two, MODEL_REGISTRY["cVAE_ConvNeXt_2D"].build_configuration().z_channels)
        self.assertNotEqual(two, three)


class DeclaredValidationTests(unittest.TestCase):
    def test_declared_bounds_are_enforced_with_the_parameter_path(self):
        cases = (
            (MatchingConfiguration(batch_size=0), "matching.batch_size"),
            (MatchingConfiguration(batch_size=True), "matching.batch_size"),
            (MatchingConfiguration(batch_size=2.5), "matching.batch_size"),
            (MatchingConfiguration(routine="nearest"), "matching.routine"),
            (MatchingConfiguration(distance_weight=float("inf")), "matching.distance_weight"),
            (MatchingConfiguration(reuse_synthetic_across_hybrids=1),
             "matching.reuse_synthetic_across_hybrids"),
            (TrainingConfiguration(validation_ratio=1.0), "training.validation_ratio"),
            (TrainingConfiguration(gradient_clip_norm=0.0), "training.gradient_clip_norm"),
            (TrainingConfiguration(early_stopping={"patience": -1}),
             "training.early_stopping"),
            (TrainingConfiguration(lr_scheduler={"factor": 1.0}), "training.lr_scheduler"),
        )
        for config, path in cases:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, path):
                config.validate()

    def test_valid_edge_values_and_extra_scheduler_arguments_pass(self):
        MatchingConfiguration(batch_size=1, distance_weight=0).validate()
        TrainingConfiguration(
            validation_ratio=0.0,
            gradient_clip_norm=None,
            trial_selection=3,
            lr_scheduler={"patience": 0, "factor": 0.5, "cooldown": 2},
        ).validate()

    def test_nested_and_sequence_parameters_are_checked(self):
        config = Configuration("nested", study_folder="results/nested")
        config.generation.feedback.max_attempts = 0
        with self.assertRaisesRegex(ValueError, r"generation\.feedback\.max_attempts"):
            config.generation.validate()

        config = Configuration("sequence", study_folder="results/sequence")
        config.extraction.roi.min_padding = (10, -1, 10)
        with self.assertRaisesRegex(ValueError, r"extraction\.roi\.min_padding\[1\]"):
            config.extraction.validate()
        config.extraction.roi.min_padding = (10, 10, 10)
        config.extraction.anomaly_size = (64, 64)
        with self.assertRaisesRegex(ValueError, r"extraction\.anomaly_size"):
            config.extraction.validate()

    def test_model_parameters_are_checked_against_their_declaration(self):
        config = Configuration("model", study_folder="results/model")
        config.model.parameters.skip_alpha = 1.5
        with self.assertRaisesRegex(ValueError, r"model\.parameters\.skip_alpha"):
            config.validate()
        config.model.parameters.skip_alpha = 0.5
        config.model.parameters.skip_alphas = [0.1, 0.2, 2.0, 0.4]
        with self.assertRaisesRegex(ValueError, r"model\.parameters\.skip_alphas\[2\]"):
            config.validate()

    def test_cross_field_rules_remain_in_validate(self):
        with self.assertRaisesRegex(ValueError, "At least one matching weight"):
            MatchingConfiguration(intensity_weight=0, gradient_weight=0).validate()
        config = Configuration("cross", study_folder="results/cross")
        config.postprocessing.rules.mask_area_fraction.minimum = 0.8
        config.postprocessing.rules.mask_area_fraction.maximum = 0.2
        with self.assertRaisesRegex(ValueError, "mask_area_fraction"):
            config.postprocessing.validate()


if __name__ == "__main__":
    unittest.main()
