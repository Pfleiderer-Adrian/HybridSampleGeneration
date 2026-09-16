"""Tests for the explicit MVTec AD 2 pipeline step API."""

import unittest

from examples.mvtec_ad2.steps import DEFAULT_GENERATION_STEPS, normalize_steps


class MVTecPipelineStepTests(unittest.TestCase):
    def test_fresh_generation_uses_default_steps(self):
        steps = normalize_steps(None)
        self.assertEqual(steps, DEFAULT_GENERATION_STEPS)
        self.assertEqual(steps, (
            "ingest", "extract", "train_generator", "generate_synthetic",
            "plan", "materialize",
        ))

    def test_existing_synthetic_anomalies_need_no_load_step(self):
        self.assertEqual(normalize_steps(("plan", "materialize")), ("plan", "materialize"))

    def test_existing_hybrid_plan_needs_no_load_step(self):
        self.assertEqual(normalize_steps("materialize"), ("materialize",))

    def test_export_and_downstream_steps_are_explicit(self):
        steps = (*DEFAULT_GENERATION_STEPS, "export", "train_downstream", "evaluate_downstream")
        self.assertEqual(normalize_steps(steps), steps)

    def test_invalid_step_types_and_empty_selection(self):
        with self.assertRaises(ValueError):
            normalize_steps(())
        with self.assertRaises(TypeError):
            normalize_steps((None,))

    def test_removed_aliases_are_rejected(self):
        for alias in ("all", "train", "generate_synth", "save", "save_config"):
            with self.subTest(alias=alias), self.assertRaises(ValueError):
                normalize_steps(alias)

    def test_conflicting_duplicate_and_unordered_steps_are_rejected(self):
        for steps in (("train_generator", "load_generator"), ("export", "export"),
                      ("materialize", "plan"), ()):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                normalize_steps(steps)


if __name__ == "__main__":
    unittest.main()
