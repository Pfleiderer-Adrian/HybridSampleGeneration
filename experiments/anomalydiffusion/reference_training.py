"""Original Lightning training/validation and checkpoint selection with V3 data."""
import json
import math
from pathlib import Path
import random
import time
import types

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from .prepare import sha256, write_json
from .runtime import load_model


MONITOR = "val/loss_simple_ema"


def validate_partitions(train_rows, validation_rows, split):
    train = {r["sample_id"] for r in split["partitions"]["train"]}
    held_out = {r["sample_id"] for r in split["partitions"]["validation"]}
    if train & held_out:
        raise ValueError("Training and validation IDs overlap.")
    if (not train_rows or not validation_rows
            or not {r["sample_id"] for r in train_rows} <= train
            or not {r["sample_id"] for r in validation_rows} <= held_out):
        raise ValueError("Generator rows do not match the frozen V3 partitions.")


class NativePatches(Dataset):
    def __init__(self, rows, stage, category, repeats=1):
        self.rows, self.stage, self.category, self.repeats = rows, stage, category, repeats

    def __len__(self):
        return len(self.rows) * self.repeats

    def __getitem__(self, index):
        from ldm.data.personalized import imagenet_templates_small
        row = self.rows[index % len(self.rows)]
        mask = (np.array(Image.open(row["mask_path"]).convert("L")) > 0).astype(np.float32)
        if self.stage == "mask":
            image = np.repeat(mask[..., None], 3, axis=2) * 2 - 1
        else:
            image = np.array(Image.open(row["image_path"]).convert("RGB"), dtype=np.float32) / 127.5 - 1
        result = {"image": image, "caption": random.choice(imagenet_templates_small).format("*")}
        if self.stage == "appearance":
            result.update(mask=mask, name=f"{self.category}+bad")
        return result


def collate_patches(rows):
    # Upstream get_input squeezes all singleton axes. Duplicate a singleton tail
    # and explicitly retain its logical weight in epoch-mean validation logging.
    count = len(rows)
    if count == 1:
        rows = rows + rows
    batch = default_collate(rows)
    batch["_logical_batch_size"] = count
    return batch


def attach_reference_hooks(model, metadata):
    """Keep upstream hooks, retaining full Lightning state alongside its exports."""
    save_original = model.on_save_checkpoint
    log_original = model.log_dict
    model._hsg_logical_batch_size = None

    def save_checkpoint(self, checkpoint):
        complete = dict(checkpoint)
        save_original(checkpoint)  # Original embedding snapshots and spatial_encoder.pt.
        checkpoint.clear()
        checkpoint.update(complete)  # Upstream clears this for frozen bases; preserve resume/best.
        checkpoint["hsg_protocol"] = metadata
        checkpoint["hsg_actual_step"] = min(int(checkpoint["global_step"]), metadata["total_steps"])
        checkpoint["hsg_rng"] = {"python": random.getstate(), "numpy": np.random.get_state(),
                                  "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}

    def log_dict(self, dictionary, *args, **kwargs):
        if self._hsg_logical_batch_size is not None:
            kwargs.setdefault("batch_size", self._hsg_logical_batch_size)
        return log_original(dictionary, *args, **kwargs)

    model.on_save_checkpoint = types.MethodType(save_checkpoint, model)
    model.log_dict = types.MethodType(log_dict, model)

    def load_checkpoint(self, checkpoint):
        if "hsg_rng" in checkpoint:
            rng = checkpoint["hsg_rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
    model.on_load_checkpoint = types.MethodType(load_checkpoint, model)


class Progress(Callback):
    def __init__(self, output, stage, steps):
        self.output, self.stage, self.steps = output, stage, steps
        self.started = time.monotonic()

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        model._hsg_logical_batch_size = batch["_logical_batch_size"]

    def on_validation_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx=0):
        model._hsg_logical_batch_size = batch["_logical_batch_size"]

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        step = trainer.global_step + 1  # PL 1.5 increments after this callback.
        if step % 100 == 1 or step == self.steps:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError("Non-finite training loss.")
            write_json(self.output / "progress.json", {"step": step, "loss": value,
                        "elapsed_seconds": time.monotonic() - self.started,
                        "epoch": trainer.current_epoch})
            print(f"{self.stage} step={step}/{self.steps} loss={value:.7f}", flush=True)

    def on_validation_end(self, trainer, model):
        values = {key: float(value.detach()) for key, value in trainer.callback_metrics.items()
                  if key.startswith("val/") and isinstance(value, torch.Tensor)}
        if MONITOR in values and not math.isfinite(values[MONITOR]):
            raise RuntimeError("Non-finite validation loss.")
        if not trainer.sanity_checking:
            record = {"step": trainer.global_step, "epoch": trainer.current_epoch, **values}
            with (self.output / "validation_history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            write_json(self.output / "validation_latest.json", record)


def train_reference(args):
    from .run import read_inputs, seed_all
    root, patches, protocol = read_inputs(args.inputs)
    validation = json.loads((root / "validation_patches.json").read_text())
    split = json.loads((root / "split_manifest.json").read_text())
    rows = [r for r in patches["rows"] if r["label"] != "good"]
    validate_partitions(rows, validation["rows"], split)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(protocol["seed"])
    pl.seed_everything(protocol["seed"], workers=True)
    model, metadata = load_model(args.stage, args.checkpoint, output / "work", patches["category"], args.latent_mask_policy)
    model.learning_rate *= args.batch_size
    steps = args.steps if args.steps is not None else protocol[f"{args.stage}_steps"]
    interval = args.save_every or (2000 if args.stage == "appearance" else 1000)
    repeats = 20 if args.stage == "appearance" else 2000
    if args.smoke:
        repeats = 1
    train_dataset = NativePatches(rows, args.stage, patches["category"], repeats)
    val_dataset = NativePatches(validation["rows"], args.stage, patches["category"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_patches)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=args.stage == "appearance", num_workers=args.num_workers,
                            collate_fn=collate_patches)
    # V3 patch datasets differ in epoch length from MVTec AD. Validate at least
    # every checkpoint interval so mask training cannot finish without validation.
    val_interval = min(interval, len(train_loader))
    while len(train_loader) % val_interval:
        val_interval -= 1
    metadata.update(stage=args.stage, training_engine="reference_lightning_v1",
        split_fingerprint=patches["split_fingerprint"], seed=protocol["seed"],
        training_patches_sha256=sha256(root / "training_patches.json"),
        validation_patches_sha256=sha256(root / "validation_patches.json"),
        base_checkpoint_sha256=sha256(args.checkpoint), total_steps=steps,
        batch_size=args.batch_size, learning_rate=model.learning_rate, smoke_test=bool(args.smoke),
        monitor=MONITOR, mode="min", use_ema=False, checkpoint_every_steps=interval,
        validation_interval_batches=val_interval, train_repeats=repeats,
        validation_policy="original validation_step on complete frozen V3 validation defect patches",
        training_sample_ids=sorted({r["sample_id"] for r in rows}),
        validation_sample_ids=sorted({r["sample_id"] for r in validation["rows"]}))
    write_json(output / "metadata.json", metadata)
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu")
        if "hsg_protocol" not in previous:
            raise ValueError("Legacy checkpoints lack reference validation history; restart for a complete best-model selection.")
        for key in ("stage", "split_fingerprint", "base_checkpoint_sha256", "latent_mask_policy",
                    "training_patches_sha256", "validation_patches_sha256", "smoke_test", "batch_size"):
            if previous["hsg_protocol"][key] != metadata[key]:
                raise ValueError(f"Reference resume protocol mismatch: {key}")
        if previous["hsg_actual_step"] >= steps:
            raise ValueError("Reference resume checkpoint already reached the requested step budget.")
        del previous
    attach_reference_hooks(model, metadata)
    checkpoint = ModelCheckpoint(dirpath=str(output / "checkpoints"), filename="{epoch:06}",
        monitor=MONITOR, mode="min", save_top_k=1, save_last=True, every_n_train_steps=interval)
    progress = Progress(output, args.stage, steps)
    trainer = pl.Trainer(gpus=1, max_steps=steps, max_epochs=-1, benchmark=True,
        callbacks=[progress, checkpoint], logger=CSVLogger(str(output), name="lightning_logs"),
        val_check_interval=val_interval, num_sanity_val_steps=2, log_every_n_steps=50,
        enable_progress_bar=False, enable_model_summary=False,
        resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Report the final validation too. Normal best selection remains with the
    # reference ModelCheckpoint callback, using the latest logged validation.
    metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)[0]
    final_score = float(metrics[MONITOR])
    if checkpoint.best_model_score is None:
        best_path = output / "checkpoints/final_best.ckpt"
        trainer.save_checkpoint(str(best_path))
        checkpoint.best_model_path = str(best_path)
        checkpoint.best_model_score = torch.tensor(final_score)
    final_path = output / "checkpoints/last.ckpt"
    trainer.save_checkpoint(str(final_path))
    write_json(output / "selection.json", {"monitor": MONITOR, "mode": "min",
        "best_checkpoint": checkpoint.best_model_path,
        "best_score": float(checkpoint.best_model_score), "last_checkpoint": str(final_path),
        "training_steps": trainer.global_step, "final_validation": metrics})
    write_json(output / "completed.json", {"steps": trainer.global_step,
        "elapsed_seconds": time.monotonic() - progress.started,
        "best_checkpoint": checkpoint.best_model_path, "best_score": float(checkpoint.best_model_score)})
