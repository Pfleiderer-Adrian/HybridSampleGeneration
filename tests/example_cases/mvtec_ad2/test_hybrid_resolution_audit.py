"""Resolution audit reconstruction and exact historical hybrid selections."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.downstream.datasets import HybridPairs
from examples.mvtec_ad2.splits import create_manifest
from experiments.anomalydiffusion.audit_hybrids import expected_placement, mask_roundtrip
from hybrid_sample_generator.fusion.classical.configuration import Config
from hybrid_sample_generator.fusion.spatial import prepare_spatial_fusion
from .test_downstream import create_images, generator_config, seed_hybrid


class HybridResolutionAuditTests(unittest.TestCase):
    def test_small_native_mask_is_padded_without_scaling(self):
        mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
        self.assertEqual(mask_roundtrip(mask, (64, 64)), (1., 4, 0))

    def test_thin_component_can_vanish_under_nearest_resize(self):
        mask = np.ones((1, 1024), dtype=bool)
        scale, count, lost = mask_roundtrip(mask, (64, 64))
        self.assertEqual(scale, 1 / 16)
        self.assertEqual(count, 0)
        self.assertEqual(lost, 1024)

    def test_placement_reconstruction_matches_actual_fusion_geometry(self):
        mask = np.zeros((1, 16, 16), dtype=np.uint8)
        mask[:, 5:11, 6:12] = 1
        control = np.zeros((3, 48, 64), np.float32)
        anomaly = np.repeat(mask.astype(np.float32), 3, axis=0)
        for scale in ((1., 1.), (.5, .5), (.8421, .8421)):
            for position in ((.5, .5), (0., 0.), (1., 1.)):
                meta = {"scale_factor": scale, "norm_type": None}
                actual = prepare_spatial_fusion(control, anomaly, meta, position, mask,
                    params=Config(fusion_keep_bg=False), spatial_ndim=2, backend_name="audit-test")
                slices, support = expected_placement(mask[0] > 0, meta, position, control.shape[-2:])
                self.assertEqual(slices, actual.output_slices)
                np.testing.assert_array_equal(support, actual.target_mask[actual.crop_to_background] > 0)

    def test_historical_selection_controls_membership_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_images(root)
            manifest = create_manifest(root / "can")
            repository, store = seed_hybrid(generator_config(root / "study"), manifest)
            original = repository.get_hybrid_sample("hybrid")
            repository.upsert_hybrid_sample(replace(original, id="second", variant_index=1))
            first_placement = repository.list_placements("hybrid")[0]
            repository.upsert_placement(replace(first_placement, id="second-placement", hybrid_sample_id="second"))
            data = DownstreamConfiguration().data
            data.patch_size = (64, 64)
            pairs = HybridPairs(repository, store, manifest, data, included_ids=["second", "hybrid"])
            self.assertEqual([r["hybrid_id"] for r in pairs.provenance], ["second", "hybrid"])
            self.assertEqual(len(HybridPairs(repository, store, manifest, data, included_ids=["second"])), 1)
            for invalid in ([], ["hybrid", "hybrid"], ["missing"]):
                with self.assertRaises(ValueError):
                    HybridPairs(repository, store, manifest, data, included_ids=invalid)
