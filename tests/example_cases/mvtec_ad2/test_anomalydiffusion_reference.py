"""Reference-runner invariants; run in the isolated AnomalyDiffusion environment."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


@unittest.skipUnless(importlib.util.find_spec("pytorch_lightning"), "Requires isolated AnomalyDiffusion runtime")
class ReferenceTrainingTests(unittest.TestCase):
    def test_train_validation_partition_guard(self):
        from experiments.anomalydiffusion.reference_training import validate_partitions
        split = {"partitions": {"train": [{"sample_id": "train"}], "validation": [{"sample_id": "val"}]}}
        validate_partitions([{"sample_id": "train"}], [{"sample_id": "val"}], split)
        for training, validation in (([{"sample_id": "val"}], [{"sample_id": "val"}]),
                                     ([{"sample_id": "train"}], [{"sample_id": "train"}])):
            with self.assertRaises(ValueError):
                validate_partitions(training, validation, split)

    def test_singleton_tail_keeps_one_logical_sample(self):
        from experiments.anomalydiffusion.reference_training import collate_patches
        row = {"image": np.zeros((8, 8, 3), np.float32), "mask": np.ones((8, 8), np.float32), "caption": "a photo of *"}
        batch = collate_patches([row])
        self.assertEqual(batch["_logical_batch_size"], 1)
        self.assertEqual(batch["image"].shape, (2, 8, 8, 3))
        torch.testing.assert_close(batch["mask"][0], batch["mask"][1])

    def test_original_clear_hook_does_not_erase_full_resume_state(self):
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from torch.utils.data import DataLoader, TensorDataset
        from experiments.anomalydiffusion.reference_training import attach_reference_hooks, MONITOR

        class Tiny(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
            def training_step(self, batch, batch_idx):
                return self.weight.square()
            def validation_step(self, batch, batch_idx):
                self.log(MONITOR, self.weight.square(), on_step=False, on_epoch=True, batch_size=len(batch[0]))
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=.1)
            def on_save_checkpoint(self, checkpoint):
                checkpoint.clear()

        with tempfile.TemporaryDirectory() as tmp:
            model = Tiny()
            attach_reference_hooks(model, {"total_steps": 3})
            callback = ModelCheckpoint(dirpath=tmp, monitor=MONITOR, mode="min", save_top_k=1,
                                       save_last=True, every_n_train_steps=1)
            trainer = pl.Trainer(gpus=0, max_steps=3, max_epochs=-1, logger=False,
                callbacks=[callback], enable_progress_bar=False, enable_model_summary=False,
                val_check_interval=1, num_sanity_val_steps=0)
            loader = DataLoader(TensorDataset(torch.zeros(8, 1)), batch_size=2)
            trainer.fit(model, loader, loader)
            self.assertTrue(Path(callback.best_model_path).is_file())
            checkpoint = torch.load(Path(tmp) / "last.ckpt", map_location="cpu")
            self.assertIn("weight", checkpoint["state_dict"])
            self.assertTrue(checkpoint["optimizer_states"][0]["state"])
            self.assertIn("hsg_rng", checkpoint)
            self.assertEqual(checkpoint["hsg_actual_step"], 3)
            self.assertLess(float(callback.best_model_score), 1.)
