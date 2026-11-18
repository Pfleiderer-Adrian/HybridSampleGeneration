"""Generate AnomalyDiffusion pairs on healthy training images (2x profile, AAR).

Masks and healthy backgrounds come from a materialized hybrid study. For every
pair one mask component is selected, a native 256 crop is taken around it and
each covering 128 window is inpainted at 2x. The result is a ``manifest.json``
read by :class:`~.pairs.AnomalyDiffusionPairs`.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import SCALE, WINDOW, stitch, windows
from .hybrids import control_and_mask, hybrid_sources
from .prepare import read_inputs, sha256, write_json
from .runtime import load_model
from .training import SEED, seed_all

COUNT = 1000
GENERATION_SEED = 62000
DDIM_STEPS = 200


def write_pair(output, index, image, healthy, mask, meta, coords):
    """Save one generated pair below ``output/pairs`` and return its manifest row."""
    prefix = Path(output) / "pairs" / f"{index:07d}"
    paths = {key: f"{prefix}_{key}.png" for key in ("image", "target", "mask")}
    Image.fromarray(image).save(paths["image"])
    Image.fromarray(healthy).save(paths["target"])
    Image.fromarray(mask.astype("uint8") * 255).save(paths["mask"])
    return dict(image_path=paths["image"], target_path=paths["target"], mask_path=paths["mask"], **meta,
                mask_source="hybrid_study", native_windows_xy=coords, native_mask_pixels=int(mask.sum()),
                file_sha256={key: sha256(path) for key, path in paths.items()})


def write_manifest(output, rows, split, patches, selection, hybrid_study, count, elapsed_seconds=0.):
    """Manifest read by the downstream loader ``AnomalyDiffusionPairs``."""
    path = Path(output) / "manifest.json"
    write_json(path, dict(
        version=1, source="anomalydiffusion", split_fingerprint=split["fingerprint"], rows=rows,
        training_sample_ids=sorted(r["sample_id"] for r in split["partitions"]["train"]),
        appearance_training_sample_ids=sorted({r["sample_id"] for r in patches["rows"]}),
        appearance_checkpoint=selection, hybrid_study=str(Path(hybrid_study).resolve()),
        generation=dict(profile="scale2x_last50k_aar", adaptive_attention=True, scale=SCALE, window=WINDOW,
                        ddim_steps=DDIM_STEPS, ddim_eta=1., batch_size=4, count=count),
        smoke_test=False, seed=GENERATION_SEED, elapsed_seconds=elapsed_seconds))
    return path


def generate(inputs, checkpoint, hybrid_study, output=None, *, training=None, hybrid_ids=None, count=COUNT):
    import torch

    root, patches, _, split = read_inputs(inputs)
    category = json.loads((root / "protocol.json").read_text())["category"]
    selection = json.loads((Path(training or root / "training") / "selection.json").read_text())
    if sha256(selection["checkpoint"]) != selection["checkpoint_sha256"]:
        raise ValueError("Selected appearance checkpoint changed after training.")
    if sha256(checkpoint) != selection["base_checkpoint_sha256"]:
        raise ValueError("Base checkpoint differs from the one used for training.")
    output = Path(output or root / "generated").resolve()
    (output / "pairs").mkdir(parents=True, exist_ok=False)
    sources = hybrid_sources(hybrid_study, split, hybrid_ids)

    seed_all(SEED)
    model, _ = load_model(checkpoint, output / "work", category, "maxpool")
    state = torch.load(selection["checkpoint"], map_location="cpu")
    expected = {k for k in model.state_dict() if k.startswith("embedding_manager.")}
    if state["step"] != selection["training_steps"] or set(state["learned_state_dict"]) != expected:
        raise ValueError("Incomplete or wrong learned appearance state.")
    model.load_state_dict(state["learned_state_dict"], strict=False)
    model.eval()
    del state

    seed_all(GENERATION_SEED)
    rng = np.random.default_rng(SEED)
    # Repeated shuffled cycles instead of favoring a handful of masks.
    order = []
    while len(order) < count:
        order.extend(rng.permutation(len(sources)).tolist())
    rows, started = [], time.monotonic()
    size = WINDOW * SCALE
    for i, index in enumerate(order[:count]):
        healthy, mask, meta = control_and_mask(hybrid_study, sources[index], rng)
        coords, generated = windows(mask), []
        for start in range(0, len(coords), 4):
            batch_coords = coords[start:start + 4]
            real_count = len(batch_coords)
            if real_count == 1:
                batch_coords = batch_coords * 2  # Upstream squeezes singleton batches.
            images, masks = [], []
            for x, y in batch_coords:
                images.append(np.asarray(Image.fromarray(healthy[y:y + WINDOW, x:x + WINDOW]).resize(
                    (size, size), Image.Resampling.BICUBIC)).astype("float32") / 127.5 - 1)
                masks.append(np.asarray(Image.fromarray(mask[y:y + WINDOW, x:x + WINDOW].astype("uint8") * 255).resize(
                    (size, size), Image.Resampling.NEAREST)) > 0)
            batch = dict(image=torch.from_numpy(np.stack(images)).cuda(),
                         mask=torch.from_numpy(np.stack(masks).astype("float32")).cuda(),
                         caption=["a photo of a *"] * len(batch_coords), name=[f"{category}+bad"] * len(batch_coords))
            with torch.no_grad():
                logs = model.log_images(batch, N=len(batch_coords), sample=False, inpaint=True,
                                        unconditional_only=True, ddim_steps=DDIM_STEPS, ddim_eta=1.,
                                        adaptive_mask=True)
            for j, (x, y) in enumerate(batch_coords[:real_count]):
                value = logs["samples_inpainting"][j].detach().cpu().permute(1, 2, 0).numpy()
                if not np.isfinite(value).all():
                    raise ValueError("Non-finite AnomalyDiffusion output.")
                generated.append((x, y, np.clip((value + 1) * 127.5, 0, 255).round().astype("uint8")))
        rows.append(write_pair(output, i, stitch(healthy, generated), healthy, mask, meta, coords))
        write_json(output / "progress.json", dict(completed_pairs=len(rows), total_pairs=count,
                                                  elapsed_seconds=time.monotonic() - started))
    return write_manifest(output, rows, split, patches, selection, hybrid_study, count, time.monotonic() - started)
