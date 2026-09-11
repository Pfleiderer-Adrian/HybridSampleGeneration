"""Tests for immutable domain records and spatial positions."""

import unittest
from dataclasses import FrozenInstanceError

from hybrid_sample_generator.domain.records import (
    OriginalSample,
    Placement,
    RealAnomaly,
)


class DomainRecordTests(unittest.TestCase):
    def test_real_anomaly_position_matches_spatial_dimensions(self):
        anomaly_2d = RealAnomaly(
            "real-2d", "original", 0, "image", "mask", "roi", "roi-mask",
            2, None, 0.25, 0.75,
        )
        anomaly_3d = RealAnomaly(
            "real-3d", "original", 0, "image", "mask", "roi", "roi-mask",
            3, 0.1, 0.25, 0.75,
        )

        self.assertEqual(anomaly_2d.source_position, (0.25, 0.75))
        self.assertEqual(anomaly_3d.source_position, (0.1, 0.25, 0.75))

    def test_placement_position_matches_spatial_dimensions(self):
        placement_2d = Placement(
            "placement-2d", "hybrid", "synthetic", 0, 2, None, 0.4, 0.6,
        )
        placement_3d = Placement(
            "placement-3d", "hybrid", "synthetic", 0, 3, 0.2, 0.4, 0.6,
        )

        self.assertEqual(placement_2d.position, (0.4, 0.6))
        self.assertEqual(placement_3d.position, (0.2, 0.4, 0.6))

    def test_records_are_frozen_and_metadata_defaults_are_independent(self):
        first = OriginalSample(
            "first", "first.npy", "image.npy", None, 2, False, False, 0
        )
        second = OriginalSample(
            "second", "second.npy", "image.npy", None, 2, False, False, 1
        )

        self.assertIsNot(first.metadata, second.metadata)
        first.metadata["source"] = "test"
        self.assertEqual(second.metadata, {})
        with self.assertRaises(FrozenInstanceError):
            first.id = "changed"


if __name__ == "__main__":
    unittest.main()
