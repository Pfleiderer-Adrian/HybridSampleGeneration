"""Tests for anomaly matching, caching, and placement planning."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.matching.planner import plan_hybrid_samples
from hybrid_sample_generator.domain.records import OriginalSample, RealAnomaly, SyntheticAnomaly
from hybrid_sample_generator.persistence.study_repository import StudyRepository
from hybrid_sample_generator.configuration.matching import MatchingConfiguration


class MatchingSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = ArtifactStore(self.root)
        self.repository = StudyRepository(self.root / "study.sqlite")
        self.config = MatchingConfiguration(routine="local", gradient_weight=0)

    def tearDown(self):
        self.temporary.cleanup()

    def _populate(self, *, controls=1, rois=5, variants=1):
        originals = []
        for index in range(controls + 1):
            name = f"control-{index}" if index < controls else "source"
            image = np.full((1, 64, 64), index, dtype=np.float32)
            path = self.store.save_entity_array("original_samples", name, "image", image)
            originals.append(OriginalSample(
                name, name, path, None, 2, index == controls, index == controls, index,
            ))
        self.repository.replace_original_samples(originals)
        self.reals = []
        for index in range(rois):
            name = f"real-{index}"
            roi = np.full((1, 4, 4), index + 1, dtype=np.float32)
            path = self.store.save_entity_array("real_anomalies", name, "roi_image", roi)
            mask = self.store.save_entity_array(
                "real_anomalies", name, "segmentation", np.ones_like(roi),
            )
            # No roi_shape metadata: unused ROI files must still remain unloaded.
            real = RealAnomaly(name, "source", index, path, mask, path, mask, 2, None, .5, .5)
            self.repository.upsert_real_anomaly(real)
            self.reals.append(real)
            for variant in range(variants):
                self.repository.upsert_synthetic_anomaly(SyntheticAnomaly(
                    f"synthetic-{index}-{variant}", name, variant, path, mask, variant,
                ))

    @staticmethod
    def _match(template, _control, _config):
        value = float(template.intensity[0, 0])
        return value / 10, (value * 8, value * 8)

    def _plan(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return plan_hybrid_samples(self.repository, self.store, self.config)

    def _selected_reals(self, plans):
        return [
            self.repository.get_synthetic_anomaly(placement.synthetic_anomaly_id).real_anomaly_id
            for plan in plans for placement in self.repository.list_placements(plan.id)
        ]

    def test_local_stops_after_first_match_while_global_ranks_all_rois(self):
        self._populate()
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=self._match) as match:
            with patch.object(self.store, "load_array", wraps=self.store.load_array) as load:
                local_plan = self._plan()
        self.assertEqual(match.call_count, 1)
        self.assertEqual(self._selected_reals(local_plan), ["real-0"])
        loaded_paths = {call.args[0] for call in load.call_args_list}
        self.assertTrue(all(real.roi_image_path not in loaded_paths for real in self.reals[1:]))
        self.assertEqual(self.repository.count_match_candidates(), 1)

        self.config.routine = "global"
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=self._match) as match:
            global_plan = self._plan()
        self.assertEqual(match.call_count, 4)  # The first ROI is already cached by local.
        self.assertEqual(self.repository.count_match_candidates(), 5)
        self.assertEqual(self._selected_reals(global_plan), ["real-4"])

        self.config.routine = "local"
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=AssertionError("cached pair")):
            with patch.object(self.store, "load_array", side_effect=AssertionError("unneeded artifact")):
                cached_plan = self._plan()
        self.assertEqual(self._selected_reals(cached_plan), ["real-0"])
        self.assertEqual(cached_plan, local_plan)

    def test_local_continues_roi_sequence_across_hybrids_and_controls(self):
        self._populate(controls=2)
        self.config.hybrids_per_original = 2
        self.config.anomalies_per_hybrid = 2
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=self._match) as match:
            with patch.object(self.store, "load_array", wraps=self.store.load_array) as load:
                plans = self._plan()
        self.assertEqual(match.call_count, 8)
        self.assertEqual(load.call_count, 7)  # Two controls and five distinct ROIs, each loaded once.
        self.assertEqual(self._selected_reals(plans), [
            "real-0", "real-1", "real-2", "real-3",
            "real-4", "real-0", "real-1", "real-2",
        ])
        self.assertTrue(all(len(self.repository.list_placements(plan.id)) == 2 for plan in plans))

    def test_local_retries_invalid_and_overlapping_rois_and_caches_rejections(self):
        self._populate()
        self.config.anomalies_per_hybrid = 2

        def match_with_rejections(template, control, config):
            index = int(template.intensity[0, 0]) - 1
            if index == 0:
                return -2.0, None
            if index in (1, 2):
                return .5, (8.0, 8.0)
            return self._match(template, control, config)

        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=match_with_rejections) as match:
            plans = self._plan()
        self.assertEqual(match.call_count, 4)
        self.assertEqual(self._selected_reals(plans), ["real-1", "real-3"])
        cached = self.repository.list_match_candidates("control-0")
        self.assertFalse(next(pair for pair in cached if pair.real_anomaly_id == "real-0").is_valid)
        self.assertEqual(len(cached), 4)
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=AssertionError("cached pair")):
            self.assertEqual(self._selected_reals(self._plan()), ["real-1", "real-3"])

    def test_local_skips_exhausted_variants_before_matching(self):
        self._populate(controls=2, rois=3)
        self.config.hybrids_per_original = 2
        self.config.reuse_synthetic_across_hybrids = False
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=self._match) as match:
            plans = self._plan()
        self.assertEqual(match.call_count, 3)
        self.assertEqual(self._selected_reals(plans), ["real-0", "real-1", "real-2"])
        placements = self.repository.list_placements()
        self.assertEqual(len({p.synthetic_anomaly_id for p in placements}), 3)

    def test_sibling_variants_share_one_match_and_respect_sibling_setting(self):
        self._populate(rois=1, variants=2)
        self.config.anomalies_per_hybrid = 2
        self.config.allow_sibling_variants_in_same_hybrid = True
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=self._match) as match:
            plans = self._plan()
        self.assertEqual(match.call_count, 1)
        self.assertEqual(self._selected_reals(plans), ["real-0", "real-0"])
        placements = self.repository.list_placements()
        self.assertEqual(len({p.synthetic_anomaly_id for p in placements}), 2)

        self.config.allow_sibling_variants_in_same_hybrid = False
        self.config.hybrids_per_original = 2
        self.config.reuse_synthetic_across_hybrids = False
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=AssertionError("cached pair")):
            plans = self._plan()
        self.assertEqual(len(plans), 2)
        self.assertTrue(all(len(self.repository.list_placements(plan.id)) == 1 for plan in plans))
        self.assertEqual(len({p.synthetic_anomaly_id for p in self.repository.list_placements()}), 2)

    def test_local_exhausts_invalid_pool_once_and_returns_no_plan(self):
        self._populate()
        self.config.hybrids_per_original = 3
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", return_value=(-2.0, None)) as match:
            plans = self._plan()
        self.assertEqual(plans, [])
        self.assertEqual(match.call_count, 5)
        self.assertEqual(self.repository.count_match_candidates(), 5)

    def test_batchwise_only_matches_configured_subset_and_ranks_it(self):
        self._populate()
        self.config.routine = "batchwise"
        self.config.batch_size = 2
        with patch("hybrid_sample_generator.matching.pair_matcher.template_matching_prepared", side_effect=self._match) as match:
            plans = self._plan()
        self.assertEqual(match.call_count, 2)
        best_score = max(self._match(*call.args)[0] for call in match.call_args_list)
        self.assertEqual(self.repository.list_placements(plans[0].id)[0].score, best_score)


if __name__ == "__main__":
    unittest.main()
