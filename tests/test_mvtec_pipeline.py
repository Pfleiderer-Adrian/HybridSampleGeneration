import unittest

from use_cases.MVTecAD2.MVTecAD2_pipeline import _default_generation_steps


class MVTecPipelineStepTests(unittest.TestCase):
    def test_existing_synthetic_anomalies_need_no_load_step(self):
        steps = _default_generation_steps(
            train_generator=False,
            load_existing_generator=False,
            generate_synthetic_anomalies=False,
            plan_hybrids=True,
        )

        self.assertEqual(
            steps,
            (
                "plan_hybrid_samples",
                "materialize_hybrid_samples",
                "save_config",
            ),
        )

    def test_existing_hybrid_plan_needs_no_load_step(self):
        steps = _default_generation_steps(
            train_generator=False,
            load_existing_generator=False,
            generate_synthetic_anomalies=False,
            plan_hybrids=False,
        )

        self.assertEqual(
            steps,
            (
                "materialize_hybrid_samples",
                "save_config",
            ),
        )


if __name__ == "__main__":
    unittest.main()
