"""Exception, nesting, and CUDA random-state restoration."""
import random
import unittest
import numpy as np
import torch
from hybrid_sample_generator.randomness import seeded_random


class RandomnessTests(unittest.TestCase):
    def test_exception_and_nested_context_restore_full_cpu_state(self):
        states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with self.assertRaises(RuntimeError):
            with seeded_random(3):
                expected = (random.getstate(), np.random.get_state(), torch.get_rng_state())
                with seeded_random(9):
                    random.random()
                    np.random.random()
                    torch.rand(1)
                self.assertEqual(random.getstate(), expected[0])
                for actual, original in zip(np.random.get_state(), expected[1]):
                    np.testing.assert_equal(actual, original)
                torch.testing.assert_close(torch.get_rng_state(), expected[2])
                raise RuntimeError('failure')
        self.assertEqual(random.getstate(), states[0])
        for actual, original in zip(np.random.get_state(), states[1]):
            np.testing.assert_equal(actual, original)
        torch.testing.assert_close(torch.get_rng_state(), states[2])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_state_is_repeatable_and_restored_after_exception(self):
        states = torch.cuda.get_rng_state_all()
        with seeded_random(7):
            first = torch.rand(4, device='cuda')
        with self.assertRaises(RuntimeError):
            with seeded_random(7):
                torch.testing.assert_close(first, torch.rand(4, device='cuda'))
                raise RuntimeError('failure')
        for actual, original in zip(torch.cuda.get_rng_state_all(), states):
            torch.testing.assert_close(actual, original)
