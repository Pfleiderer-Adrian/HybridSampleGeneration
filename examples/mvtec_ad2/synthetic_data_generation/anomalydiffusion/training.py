"""Appearance training with the original Lightning module (2x profile, last checkpoint).

Only the learned text embeddings and the spatial encoder are trained; the base
latent diffusion model stays frozen and is referenced by its checksum.
"""
import json
import math
import random
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

from .prepare import read_inputs, sha256, write_json
from .runtime import load_model

MONITOR = "val/loss_simple_ema"
STEPS = 50000
MILESTONES = (10000, 20000, 30000, 40000, 50000)
SEED = 42


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NativePatches:
    def __init__(self, rows, category, repeats=1):
        self.rows, self.category, self.repeats = rows, category, repeats

    def __len__(self):
        return len(self.rows) * self.repeats

    def __getitem__(self, index):
        from ldm.data.personalized import imagenet_templates_small
        row = self.rows[index % len(self.rows)]
        mask = (np.array(Image.open(row["mask_path"]).convert("L")) > 0).astype(np.float32)
        image = np.array(Image.open(row["image_path"]).convert("RGB"), dtype=np.float32) / 127.5 - 1
        return {"image": image, "caption": random.choice(imagenet_templates_small).format("*"),
                "mask": mask, "name": f"{self.category}+bad"}


def collate_patches(rows):
    from torch.utils.data._utils.collate import default_collate
    # Upstream get_input squeezes all singleton axes. Duplicate a singleton tail
    # and explicitly retain its logical weight in epoch-mean validation logging.
    count = len(rows)
    if count == 1:
        rows = rows + rows
    batch = default_collate(rows)
    batch["_logical_batch_size"] = count
    return batch


def attach_reference_hooks(model, metadata):
    """Keep the upstream hooks; log validation with the logical batch size."""
    log_original = model.log_dict
    model._hsg_logical_batch_size = None

    def log_dict(self, dictionary, *args, **kwargs):
        if self._hsg_logical_batch_size is not None:
            kwargs.setdefault("batch_size", self._hsg_logical_batch_size)
        return log_original(dictionary, *args, **kwargs)

    model.log_dict = types.MethodType(log_dict, model)


def train(inputs, checkpoint, output=None, *, steps=STEPS, milestones=MILESTONES):
    """Train appearance weights and select the final step as in the V3 submissions."""
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader

    root, patches, validation, split = read_inputs(inputs)
    protocol = json.loads((root / "protocol.json").read_text())
    category = protocol["category"]
    output = Path(output or root / "training").resolve()
    output.mkdir(parents=True, exist_ok=False)
    milestones = sorted(set(milestones) | {steps})
    torch.set_num_threads(4)
    seed_all(SEED)
    pl.seed_everything(SEED, workers=True)
    model, meta = load_model(checkpoint, output / "work", category, "maxpool")
    model.learning_rate *= 4  # Original batch-size scaling: effective 0.02 at batch size 4.
    meta.update(profile=protocol["profile"], total_steps=steps, seed=SEED, batch_size=4,
                learning_rate=model.learning_rate, base_checkpoint_sha256=sha256(checkpoint),
                split_fingerprint=split["fingerprint"],
                training_patches_sha256=sha256(root / "training_patches.json"),
                validation_patches_sha256=sha256(root / "validation_patches.json"))
    write_json(output / "metadata.json", {k: v for k, v in meta.items() if k not in ("missing_keys", "unexpected_keys")})
    attach_reference_hooks(model, meta)
    train_loader = DataLoader(NativePatches(patches["rows"], category, 20), batch_size=4, shuffle=True,
                              num_workers=2, collate_fn=collate_patches)
    validation_loader = DataLoader(NativePatches(validation["rows"], category), batch_size=4, shuffle=True,
                                   num_workers=2, collate_fn=collate_patches)

    class Progress(Callback):
        started = time.monotonic()

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            model._hsg_logical_batch_size = batch["_logical_batch_size"]

        def on_validation_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx=0):
            model._hsg_logical_batch_size = batch["_logical_batch_size"]

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            step = trainer.global_step + 1  # Lightning 1.5 increments after this callback.
            if step % 100 == 1 or step == steps:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite training loss.")
                write_json(output / "progress.json", {"step": step, "loss": value,
                                                      "elapsed_seconds": time.monotonic() - self.started})
                print(f"appearance step={step}/{steps} loss={value:.7f}", flush=True)

        def on_validation_end(self, trainer, model):
            metric = trainer.callback_metrics.get(MONITOR)
            if metric is not None and not math.isfinite(float(metric)):
                raise RuntimeError("Non-finite validation loss.")
            if metric is not None and not trainer.sanity_checking:
                with (output / "validation_history.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": trainer.global_step, MONITOR: float(metric)}) + "\n")

    class Compact(Callback):
        """Learned state only; the frozen base is referenced by checksum."""
        best = float("inf")

        def save(self, trainer, model, name, step, optimizer=False, score=None):
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                     if k.startswith("embedding_manager.")}
            payload = dict(format="hsg_anomalydiffusion_v1", metadata=meta, step=step,
                           learned_state_dict=state, validation_score=score)
            if optimizer:
                payload.update(optimizer_state=trainer.optimizers[0].state_dict(),
                               rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
            path = output / "checkpoints" / name
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            step = trainer.global_step + 1
            if step in milestones:
                self.save(trainer, model, f"step_{step:06d}.pt", step)
            if step % 1000 == 0:
                self.save(trainer, model, "latest_resume.pt", step, optimizer=True)

        def on_validation_end(self, trainer, model):
            if trainer.sanity_checking:
                return
            metric = trainer.callback_metrics.get(MONITOR)
            if metric is None:
                return
            score = float(metric)
            if np.isfinite(score) and score < self.best:
                self.best = score
                self.save(trainer, model, "best.pt", trainer.global_step, score=score)
                write_json(output / "best.json", dict(step=trainer.global_step, score=score, monitor=MONITOR))

    interval = min(1000, len(train_loader))
    while len(train_loader) % interval:
        interval -= 1
    trainer = pl.Trainer(gpus=1, max_steps=steps, max_epochs=-1, benchmark=True, enable_checkpointing=False,
                         callbacks=[Progress(), Compact()], logger=CSVLogger(str(output), name="lightning_logs"),
                         val_check_interval=interval, num_sanity_val_steps=2, log_every_n_steps=50,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    final = trainer.validate(model, dataloaders=validation_loader, verbose=False)[0]
    last = output / "checkpoints" / f"step_{steps:06d}.pt"
    if not last.is_file():
        raise ValueError(f"Missing final checkpoint {last.name}.")
    write_json(output / "selection.json", dict(
        selection="last", training_steps=steps, checkpoint=str(last), checkpoint_sha256=sha256(last),
        base_checkpoint_sha256=meta["base_checkpoint_sha256"], final_validation=final,
        training_sample_ids=sorted({r["sample_id"] for r in patches["rows"]})))
    return output
