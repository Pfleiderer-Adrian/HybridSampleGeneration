"""Tests for typed input samples and ingestion adapters."""

import unittest
from pathlib import Path

import numpy as np

from hybrid_sample_generator.domain.input_sample import (
    InputSample,
    coerce_input_sample,
    iter_input_samples,
)


class InputSampleTests(unittest.TestCase):
    def test_existing_input_sample_is_preserved(self):
        sample = InputSample(
            np.ones((1, 3, 4)),
            None,
            "sample",
            metadata={"group": "control"},
        )

        self.assertIs(coerce_input_sample(sample), sample)

    def test_tuple_input_is_normalized_and_keeps_source_paths(self):
        image = [[1, 2], [3, 4]]
        segmentation = [[0, 1], [0, 0]]

        sample = coerce_input_sample(
            (
                image,
                segmentation,
                Path("image.npy"),
                Path("/source/image.npy"),
                Path("/source/mask.npy"),
            )
        )

        np.testing.assert_array_equal(sample.image, np.asarray(image))
        np.testing.assert_array_equal(sample.segmentation, np.asarray(segmentation))
        self.assertEqual(sample.source_name, "image.npy")
        self.assertEqual(sample.source_image_path, "/source/image.npy")
        self.assertEqual(sample.source_segmentation_path, "/source/mask.npy")

    def test_invalid_input_shape_is_rejected(self):
        for value in (object(), ("image", "mask")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "dataloader item"):
                    coerce_input_sample(value)

    def test_typed_dataloader_protocol_takes_precedence(self):
        expected = InputSample(np.zeros((1, 2, 2)), None, "typed")

        class Dataloader:
            def __iter__(self):
                raise AssertionError("generic iteration must not be used")

            def iter_input_samples(self):
                yield expected

        self.assertEqual(list(iter_input_samples(Dataloader())), [expected])


if __name__ == "__main__":
    unittest.main()
