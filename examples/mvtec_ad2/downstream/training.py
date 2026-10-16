"""Joint DRAEM training and validation-based checkpoint selection."""

from collections import Counter
import csv
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import build_model
from .draem.losses import training_loss
from .evaluation import evaluate, evaluation_loader


@torch.no_grad()
def paired_validation_loss(model, loader, device):
    model.eval()
    total, count = 0., 0
    for batch in loader:
        image, target, mask = (batch[key].to(device) for key in ("image", "target", "mask"))
        total += model.paired_loss(image, target, mask).item() * len(image)
        count += len(image)
    if not count:
        raise ValueError("Paired validation must not be empty.")
    return total / count


def train(dataset, validation, config, output, *, paired_validation=None):
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    settings = config.training
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if settings.device == "auto" else settings.device)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sorted(set([max(1, int(settings.epochs*.8)), max(1, int(settings.epochs*.9))])), gamma=.2)
    loader = DataLoader(dataset, batch_size=settings.batch_size, num_workers=settings.num_workers,
                        persistent_workers=False,
                        generator=torch.Generator().manual_seed(config.seed))
    validation_loader = evaluation_loader(validation, settings)
    maximize = config.model == "draem" or settings.selection_metric == "pixel_aupro_0.05"
    paired_loader = None
    if config.model == "dinov3_paired" and not maximize:
        if paired_validation is None:
            raise ValueError("DINOv3 checkpoint selection requires fixed held-out validation pairs.")
        paired_loader = DataLoader(paired_validation, batch_size=settings.batch_size,
                                   num_workers=settings.num_workers, shuffle=False,
                                   generator=torch.Generator().manual_seed(config.seed + 104729))
    output = Path(output)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    best = -float("inf") if maximize else float("inf")
    with (output / "training_history.csv").open("w", newline="") as file:
        fields = ["epoch", "train_loss", "validation_loss", "selection_metric", "selection_mode", "selection_score", "selection_loss", "real", "hard_normal", "normal", "draem", "hybrid"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(settings.epochs):
            dataset.set_epoch(epoch)
            model.train()
            total, seen = 0., 0
            counts = Counter()
            hard_count = 0
            for batch in tqdm(loader, desc=f"{config.model} epoch {epoch+1}/{settings.epochs}"):
                image, target, mask = (batch[key].to(device) for key in ("image", "target", "mask"))
                optimizer.zero_grad(set_to_none=True)
                if config.model == "dinov3_paired":
                    if "real" in batch["source"]:
                        paired = torch.tensor([s != "real" for s in batch["source"]], device=device)
                        loss = image.new_zeros(())
                        if paired.any():
                            loss = model.paired_loss(image[paired], target[paired], mask[paired]) * paired.sum() / len(image)
                        if (~paired).any():
                            _, logits = model(image[~paired])
                            loss = loss + model.segmentation_loss(logits, mask[~paired]) * (~paired).sum() / len(image)
                    else:
                        loss = model.paired_loss(image, target, mask)
                elif config.model == "dinov3_segmentation":
                    _, logits = model(image)
                    loss = model.segmentation_loss(logits, mask)
                else:
                    reconstruction, logits = model(image)
                    loss = training_loss(reconstruction, logits, target, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite DRAEM training loss.")
                loss.backward()
                optimizer.step()
                total += loss.item()*len(image)
                seen += len(image)
                counts.update(batch["source"])
                if "hard_normal" in batch:
                    hard_count += int(batch["hard_normal"].sum())
            validated = (epoch + 1) % settings.validation_interval == 0 or epoch + 1 == settings.epochs
            metrics, score = {}, None
            selection_metric = ("pixel_aupro_0.05" if maximize else
                                "paired_total_loss" if paired_loader is not None else
                                "segmentation_total_loss" if config.model == "dinov3_segmentation" else
                                "segmentation_loss")
            if validated:
                metrics = evaluate(model, validation_loader, device, patch_batch_size=settings.batch_size)
                score = (paired_validation_loss(model, paired_loader, device)
                         if paired_loader is not None else metrics["segmentation_loss"])
                if config.model == "dinov3_segmentation":
                    score = metrics["segmentation_total_loss"]
                if paired_loader is not None:
                    metrics["paired_total_loss"] = score
                selection_metric = "paired_total_loss" if paired_loader is not None else "segmentation_loss"
                if config.model == "dinov3_segmentation":
                    selection_metric = "segmentation_total_loss"
                if maximize:
                    selection_metric = "pixel_aupro_0.05"
                    score = metrics.get(selection_metric)
                    if score is None:
                        raise ValueError("Checkpoint selection by AUC-PRO0.05 requires validation defect regions and background pixels.")
                if not math.isfinite(score):
                    raise FloatingPointError(f"Nonfinite validation checkpoint metric: {selection_metric}.")
            scheduler.step()
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "epoch": epoch+1,
                          "configuration": config.to_dict(), "validation": metrics}
            checkpoint["selection_metric"] = selection_metric
            checkpoint["selection_mode"] = "max" if maximize else "min"
            checkpoint["selection_score"] = score
            if not maximize:
                checkpoint["selection_loss"] = score
            if config.model in ("dinov3_paired", "dinov3_segmentation"):
                checkpoint["backbone_config"] = model.backbone.config.to_dict()
            torch.save(checkpoint, checkpoints / "last.pt")
            if validated and (score > best if maximize else score < best):
                best = score
                torch.save(checkpoint, checkpoints / "best.pt")
            writer.writerow({"epoch": epoch+1, "train_loss": total/seen,
                             "validation_loss": metrics.get("segmentation_loss", ""),
                             "selection_metric": selection_metric, "selection_mode": checkpoint["selection_mode"],
                             "selection_score": score if validated else "", "selection_loss": "" if maximize or not validated else score,
                             "hard_normal": hard_count,
                             **{source: counts[source] for source in ("normal", "draem", "hybrid", "real")}})
            file.flush()
    return output
