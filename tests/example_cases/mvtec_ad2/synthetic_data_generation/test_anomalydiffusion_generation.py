"""AnomalyDiffusion preparation, hybrid sources and manifest format (no model, no GPU)."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.mvtec_ad2.data.splits import create_manifest
from examples.mvtec_ad2.downstream.configuration import DownstreamConfiguration
from examples.mvtec_ad2.synthetic_data_generation.anomalydiffusion import geometry
from examples.mvtec_ad2.synthetic_data_generation.anomalydiffusion.generation import write_manifest, write_pair
from examples.mvtec_ad2.synthetic_data_generation.anomalydiffusion.hybrids import control_and_mask, hybrid_sources
from examples.mvtec_ad2.synthetic_data_generation.anomalydiffusion.pairs import AnomalyDiffusionPairs
from examples.mvtec_ad2.synthetic_data_generation.anomalydiffusion.prepare import prepare, read_inputs

from ..fixtures import create_images, generator_config, seed_hybrid


class GeometryTests(unittest.TestCase):
    def test_windows_cover_small_and_large_components(self):
        mask = np.zeros((256, 256), bool)
        mask[3, 250] = True
        mask[40:230, 10:200] = True
        coords = geometry.windows(mask)
        cover = np.zeros_like(mask)
        for x, y in coords:
            cover[y:y + 128, x:x + 128] = True
        self.assertTrue(cover[mask].all())
        self.assertGreater(len(coords), 2)
        with self.assertRaises(ValueError):
            geometry.windows(np.zeros((256, 256), bool))

    def test_stitch_keeps_context_outside_generated_windows(self):
        healthy = np.zeros((256, 256, 3), np.uint8)
        generated = [(0, 0, np.full((256, 256, 3), 200, np.uint8)), (64, 0, np.full((256, 256, 3), 100, np.uint8))]
        out = geometry.stitch(healthy, generated)
        self.assertEqual(out[0, 0, 0], 200)
        self.assertEqual(out[0, 100, 0], 150)  # Overlap is averaged.
        self.assertEqual(out[200, 200, 0], 0)

    def test_defect_windows_upscale_2x_and_cover_every_pixel(self):
        image = np.zeros((300, 400, 3), np.uint8)
        mask = np.zeros((300, 400), bool)
        mask[10:12, 390:400] = True
        mask[100:290, 20:250] = True
        rows = list(geometry.defect_windows(image, mask))
        self.assertTrue(all(w.shape == (256, 256, 3) and m.shape == (256, 256) for w, m, _, _ in rows))
        self.assertEqual({component for *_, component in rows}, {1, 2})


class PreparationAndSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        create_images(self.root)
        self.manifest = create_manifest(self.root / "can")
        self.split_path = self.root / "split_manifest.json"
        self.split_path.write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepare_uses_only_defects_of_train_and_validation(self):
        stats = prepare("can", self.split_path, self.root / "inputs")
        _, patches, validation, _ = read_inputs(self.root / "inputs")
        train = {s["sample_id"] for s in self.manifest["partitions"]["train"] if s["label"] != "good"}
        self.assertEqual({r["sample_id"] for r in patches["rows"]}, train)
        self.assertEqual(stats["train"]["patches"], len(patches["rows"]))
        self.assertTrue(validation["rows"])
        with self.assertRaises(FileExistsError):
            prepare("can", self.split_path, self.root / "inputs")
        # Leaking a held-out image into the training windows is rejected.
        leaked = dict(patches, rows=patches["rows"] + validation["rows"][:1])
        (self.root / "inputs/training_patches.json").write_text(json.dumps(leaked))
        with self.assertRaisesRegex(ValueError, "partitions"):
            read_inputs(self.root / "inputs")

    def test_sources_selection_and_manifest_accepted_by_downstream_loader(self):
        config = generator_config(self.root / "study")
        seed_hybrid(config, self.manifest)
        study = self.root / "study"
        sources = hybrid_sources(study, self.manifest)
        self.assertEqual([s["hybrid_id"] for s in sources], ["hybrid"])
        ids = self.root / "provenance.json"
        ids.write_text(json.dumps({"hybrid_provenance": [{"hybrid_id": "hybrid"}]}))
        self.assertEqual(hybrid_sources(study, self.manifest, ids)[0]["hybrid_id"], "hybrid")
        ids.write_text(json.dumps(["missing"]))
        with self.assertRaisesRegex(ValueError, "ineligible"):
            hybrid_sources(study, self.manifest, ids)

        # Build a manifest exactly as generate() does, with the healthy crop as "generated" output.
        prepare("can", self.split_path, self.root / "inputs")
        _, patches, _, split = read_inputs(self.root / "inputs")
        output = self.root / "generated"
        (output / "pairs").mkdir(parents=True)
        healthy, mask, meta = control_and_mask(study, sources[0], np.random.default_rng(42))
        coords = geometry.windows(mask)
        image = geometry.stitch(healthy, [(x, y, geometry.upscale_image(healthy[y:y + 128, x:x + 128]))
                                          for x, y in coords])
        rows = [write_pair(output, 0, image, healthy, mask, meta, coords)]
        path = write_manifest(output, rows, split, patches, {"selection": "last"}, study, count=1)
        data = DownstreamConfiguration().data
        data.mode, data.patch_size = "patch", (256, 256)
        pairs = AnomalyDiffusionPairs(path, self.manifest, data)
        image, target, mask = pairs.sample(0, np.random.default_rng(0))
        self.assertEqual(tuple(image.shape), (3, 256, 256))
        self.assertTrue(mask.any())

    def test_sources_skip_hybrids_excluded_by_postprocessing(self):
        config = generator_config(self.root / "study")
        seed_hybrid(config, self.manifest)
        study = self.root / "study"
        (study / "configuration.json").write_text(json.dumps({"postprocessing": {"enabled": True, "action": "exclude"}}))
        with sqlite3.connect(study / "artifacts.sqlite") as db:
            db.execute("INSERT INTO postprocessing_decisions VALUES ('hybrid','r','f','s',0,'{}','[]')")
        with self.assertRaisesRegex(ValueError, "No eligible"):
            hybrid_sources(study, self.manifest)
