"""Energy counter units and conservative assignment of GPU costs."""
import unittest

from experiments.anomalydiffusion.telemetry import allocation, interval_energy


class EnergyTelemetryTests(unittest.TestCase):
    def test_counter_delta_uses_millijoules_and_excludes_previous_usage(self):
        before = {"energy_total_mj": 900000000000, "power_w": 250.}
        after = {"energy_total_mj": 900001400000, "power_w": 300.}
        joules, method = interval_energy(before, after, 5.)
        self.assertEqual(joules, 1400.)
        self.assertEqual(method, "nvml_energy_counter")
        self.assertAlmostEqual(joules / 3.6e6, .0003888888888888889)

    def test_counter_reset_or_absence_is_explicit_power_estimate(self):
        for counter in (None, 10):
            joules, method = interval_energy({"energy_total_mj": 100, "power_w": 200.},
                                            {"energy_total_mj": counter, "power_w": 300.}, 4.)
            self.assertEqual(joules, 1000.)
            self.assertEqual(method, "power_trapezoid_fallback")

    def test_transition_and_foreign_process_are_not_assigned_to_training(self):
        first = {"job": {"category": "can", "stage": "mask_training", "pid": 1}, "unexpected_gpu_pids": []}
        self.assertEqual(allocation(first, first), "can/mask_training")
        second = {"job": {"category": "can", "stage": "appearance_training", "pid": 2}}
        self.assertEqual(allocation(first, second), "transition_unattributed")
        self.assertEqual(allocation(first, {**first, "unexpected_gpu_pids": [3]}), "interference_unattributed")
        self.assertEqual(allocation(first, {**first, "unmapped_gpu_pids": [3]}), "pid_unverified/can/mask_training")
        self.assertEqual(allocation({}, {}), "idle_unattributed")
