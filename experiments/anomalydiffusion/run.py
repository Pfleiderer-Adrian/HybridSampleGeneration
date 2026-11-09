"""Train original AD embeddings/pSp, generate masks and paired native patches."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from .prepare import latent_pixels, sha256, write_json
from .runtime import load_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_array(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1


def batch_for(rows, stage, category):
    from ldm.data.personalized import imagenet_templates_small
    values = []
    for row in rows:
        if stage == "mask":
            mask = np.array(Image.open(row["mask_path"]).convert("L")) > 0
            image = np.repeat(mask[:, :, None], 3, 2).astype(np.float32) * 2 - 1
        else:
            image = image_array(row["image_path"])
        values.append(image)
    batch = {"image": torch.from_numpy(np.stack(values)).cuda(),
             "caption": [random.choice(imagenet_templates_small).format("*") for _ in rows]}
    if stage == "appearance":
        masks = [(np.array(Image.open(row["mask_path"]).convert("L")) > 0).astype(np.float32) for row in rows]
        batch.update(mask=torch.from_numpy(np.stack(masks)).cuda(), name=[f"{category}+bad"] * len(rows))
    return batch


def read_inputs(root):
    root = Path(root).resolve()
    patches = json.loads((root / "training_patches.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    split = json.loads((root / "split_manifest.json").read_text())
    train = {r["sample_id"] for r in split["partitions"]["train"]}
    forbidden = {r["sample_id"] for p in ("validation", "test") for r in split["partitions"][p]}
    used = {r["sample_id"] for r in patches["rows"]}
    if not used or not used <= train or used & forbidden or patches["split_fingerprint"] != split["fingerprint"]:
        raise ValueError("Prepared generator sources are not training-only.")
    patches["manifest_sha256"] = sha256(root / "training_patches.json")
    return root, patches, protocol


def train(args):
    from .reference_training import train_reference
    return train_reference(args)


def learned_model(stage, path, args, patches, work):
    if Path(path).suffix == ".json":
        selection = json.loads(Path(path).read_text())
        path = Path(selection["best_checkpoint"])
    state = torch.load(path, map_location="cpu")
    reference = "hsg_protocol" in state
    meta = state["hsg_protocol"] if reference else state["metadata"]
    if reference:
        selection = json.loads((Path(path).parent.parent / "selection.json").read_text())
        if (selection["training_steps"] != meta["total_steps"]
                or Path(selection["best_checkpoint"]).resolve() != Path(path).resolve()):
            raise ValueError("Generation requires the validation-selected checkpoint of a completed reference run.")
        meta = dict(meta)
        meta.update(selected_checkpoint=str(Path(path).resolve()), selected_step=state["hsg_actual_step"],
                    selected_checkpoint_sha256=sha256(path), selected_validation_score=selection["best_score"])
    if (meta["stage"] != stage or meta["split_fingerprint"] != patches["split_fingerprint"]
            or meta["training_patches_sha256"] != patches["manifest_sha256"]
            or meta["latent_mask_policy"] != args.latent_mask_policy
            or meta["base_checkpoint_sha256"] != sha256(args.checkpoint)
            or (not reference and state["step"] != meta["total_steps"])):
        raise ValueError("Learned checkpoint does not match this experiment or is incomplete.")
    if meta["smoke_test"] and not args.smoke:
        raise ValueError("Smoke-test weights cannot produce a benchmark cache.")
    model, _ = load_model(stage, args.checkpoint, work, patches["category"], args.latent_mask_policy)
    if reference:
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        model.embedding_manager.load_state_dict(state["embedding"], strict=True)
    return model.eval(), meta


def generate(args):
    root, patches, protocol = read_inputs(args.inputs)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(protocol["seed"])
    mask_model, mask_meta = learned_model("mask", args.mask_weights, args, patches, output / "mask_work")
    defects = [r for r in patches["rows"] if r["label"] != "good"]
    masks = []
    attempts = 0
    with torch.no_grad():
        while len(masks) < args.count and attempts < args.count * 20:
            batch = batch_for([defects[0], defects[0]], "mask", patches["category"])
            logs = mask_model.log_images(batch, N=2, sample=True, inpaint=False,
                        unconditional_only=True, ddim_steps=args.mask_ddim_steps, ddim_eta=1.)
            for value in logs["samples_scaled"]:
                attempts += 1
                mask = (value.mean(0).cpu().numpy() > .8).astype(np.uint8) * 255
                # Upstream edge filter plus explicit nonempty/latent-occupancy checks.
                if not mask.any() or any(np.count_nonzero(edge) > 256 / 3 for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1])):
                    continue
                if args.latent_mask_policy == "original" and not latent_pixels(mask):
                    continue
                masks.append(mask)
                if len(masks) == args.count:
                    break
    if len(masks) < args.count:
        raise RuntimeError(f"Only {len(masks)}/{args.count} usable generated masks after {attempts} attempts. No real-mask fallback.")
    del mask_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model, appearance_meta = learned_model("appearance", args.appearance_weights, args, patches, output / "appearance_work")
    from ldm.data.personalized import imagenet_templates_small
    healthy = [r for r in patches["rows"] if r["label"] == "good"]
    rng = np.random.default_rng(protocol["seed"])
    rows = []
    pair_folder = output / "pairs"
    pair_folder.mkdir()
    with torch.no_grad():
        for start in range(0, args.count, 2):
            indices = list(range(start, min(start + 2, args.count)))
            # Avoid upstream squeeze() dropping the batch dimension for the last sample.
            if len(indices) == 1:
                indices.append(indices[0])
            backgrounds = [healthy[int(rng.integers(len(healthy)))] for _ in indices]
            batch = {"image": torch.from_numpy(np.stack([image_array(r["image_path"]) for r in backgrounds])).cuda(),
                     "mask": torch.from_numpy(np.stack([masks[i] / 255 for i in indices]).astype(np.float32)).cuda(),
                     "caption": [random.choice(imagenet_templates_small).format("*") for _ in indices],
                     "name": [f"{patches['category']}+bad"] * 2}
            logs = model.log_images(batch, N=2, sample=False, inpaint=True,
                       unconditional_only=True, ddim_steps=args.ddim_steps, ddim_eta=1., adaptive_mask=args.adaptive_attention)
            for offset, index in enumerate(indices):
                if index < len(rows):
                    continue
                name = f"{index:07d}"
                value = logs["samples_inpainting"][offset].detach().cpu().permute(1, 2, 0).numpy()
                if not np.isfinite(value).all():
                    raise ValueError("Non-finite generated image.")
                image = np.clip((value + 1) * 127.5, 0, 255).round().astype(np.uint8)
                recipient = backgrounds[offset]
                Image.fromarray(image).save(pair_folder / f"{name}_image.png")
                Image.open(recipient["image_path"]).save(pair_folder / f"{name}_target.png")
                Image.fromarray(masks[index]).save(pair_folder / f"{name}_mask.png")
                rows.append({"image_path": str(pair_folder / f"{name}_image.png"),
                             "target_path": str(pair_folder / f"{name}_target.png"),
                             "mask_path": str(pair_folder / f"{name}_mask.png"),
                             "recipient_sample_id": recipient["sample_id"],
                             "crop_xywh": recipient["crop_xywh"], "mask_source": "learned_anomalydiffusion"})
    write_json(output / "manifest.json", {"version": 1, "source": "anomalydiffusion",
            "split_fingerprint": patches["split_fingerprint"], "rows": rows,
            "training_sample_ids": sorted({r["sample_id"] for r in patches["rows"]}),
            "mask_training": mask_meta, "appearance_training": appearance_meta,
            "mask_weights_sha256": mask_meta.get("selected_checkpoint_sha256") or sha256(args.mask_weights),
            "appearance_weights_sha256": appearance_meta.get("selected_checkpoint_sha256") or sha256(args.appearance_weights),
            "adaptive_attention": args.adaptive_attention, "ddim_steps": args.ddim_steps,
            "mask_ddim_steps": args.mask_ddim_steps,
            "seed": protocol["seed"], "smoke_test": args.smoke,
            "mask_attempts": attempts, "latent_mask_policy": args.latent_mask_policy})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "generate"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-mask-policy", choices=("maxpool", "original"), default="maxpool")
    parser.add_argument("--stage", choices=("mask", "appearance"), default="appearance")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mask-weights", type=Path)
    parser.add_argument("--appearance-weights", type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=200)
    parser.add_argument("--mask-ddim-steps", type=int, default=100)
    parser.add_argument("--adaptive-attention", action="store_true")
    args = parser.parse_args()
    for key in ("inputs", "output", "checkpoint", "mask_weights", "appearance_weights", "resume"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.batch_size < 2 or (args.save_every is not None and args.save_every < 1) or (args.steps is not None and args.steps < 1) or args.count < 1 or args.num_workers < 0:
        parser.error("Batch size must be >=2; steps/count/save-every must be positive.")
    for value in (args.ddim_steps, args.mask_ddim_steps):
        if value < 1 or value > 500 or 1000 % value:
            parser.error("DDIM steps must be a positive divisor of 1000, at most 500.")
    if args.action == "generate" and (not args.mask_weights or not args.appearance_weights):
        parser.error("Generation requires both learned mask and appearance weights.")
    if args.smoke and args.action == "train" and (args.steps is None or args.steps > 10):
        parser.error("Smoke training requires explicit --steps between 1 and 10.")
    if not torch.cuda.is_available():
        parser.error("The original AnomalyDiffusion implementation requires CUDA.")
    train(args) if args.action == "train" else generate(args)


if __name__ == "__main__":
    main()
