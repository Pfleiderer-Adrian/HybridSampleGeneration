"""Real-image evaluation; bounded-memory pixel curves and saved predictions."""

import csv
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from examples.common.image_io import save_image

from .draem.losses import focal_loss
from .patches import tiled_logits


def evaluation_loader(dataset, settings):
    # Full-resolution images may have different shapes; tile one image at a time.
    return DataLoader(dataset, batch_size=1 if dataset.data.mode == "patch" else settings.batch_size,
                      num_workers=settings.num_workers)


def binary_metrics(labels, scores):
    """Exact threshold grouping, including tied predictions."""
    labels, scores = np.asarray(labels, dtype=np.int64), np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    endpoints = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.cumsum(y)[endpoints]
    fp = (endpoints + 1) - tp
    return curve_metrics(tp, fp, int(y.sum()), int(len(y)-y.sum()))


def curve_metrics(tp, fp, positives, negatives):
    if positives == 0 or negatives == 0:
        return {"auroc": None, "ap": None}
    recall = np.r_[0., tp/positives]
    fpr = np.r_[0., fp/negatives]
    precision = tp / np.maximum(tp+fp, 1)
    return {"auroc": float(np.sum(np.diff(fpr)*(recall[1:]+recall[:-1])/2)),
            "ap": float(np.sum(np.diff(recall)*precision))}


@torch.no_grad()
def evaluate(model, loader, device, output=None, bins=4096, *, patch_batch_size=1):
    model.eval()
    positive = np.zeros(bins, dtype=np.int64)
    negative = np.zeros(bins, dtype=np.int64)
    labels, scores, rows = [], [], []
    total_loss, count = 0., 0
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        data = loader.dataset.data
        if data.mode == "patch":
            images, masks = batch["image"], batch["mask"]
            logits = torch.stack([tiled_logits(model, image, device, data.patch_size,
                                              data.patch_overlap, patch_batch_size) for image in images])
        else:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            _, logits = model(images)
        total_loss += focal_loss(logits, masks).item() * len(images)
        count += len(images)
        maps = logits.softmax(1)[:, 1:2]
        # Same 21x21 mean-pooling image score convention as DRAEM.
        image_scores = F.avg_pool2d(maps, 21, stride=1, padding=10).flatten(1).amax(1).cpu().numpy()
        probabilities = maps[:, 0].cpu().numpy()
        truth = masks[:, 0].cpu().numpy() > 0
        bucket = np.clip((probabilities*(bins-1)).astype(np.int64), 0, bins-1)
        positive += np.bincount(bucket[truth], minlength=bins)
        negative += np.bincount(bucket[~truth], minlength=bins)
        labels.extend(batch["label"].tolist())
        scores.extend(image_scores.tolist())
        for sample_id, label, score, probability in zip(batch["sample_id"], batch["label"], image_scores, probabilities):
            rows.append({"sample_id": sample_id, "label": int(label), "score": float(score)})
            if output:
                # Sequential filenames avoid treating external sample IDs as paths.
                np.save(output / f"{len(rows)-1:06d}.npy", probability)
                if len(rows) <= 16:
                    save_image(probability[None], output / f"{len(rows)-1:06d}.png")
    if not count:
        raise ValueError("Cannot evaluate an empty split.")
    image = binary_metrics(labels, scores)
    pixel = curve_metrics(np.cumsum(positive[::-1]), np.cumsum(negative[::-1]), int(positive.sum()), int(negative.sum()))
    if output:
        with (output / "index.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["sample_id", "label", "score"])
            writer.writeheader()
            writer.writerows(rows)
    return {"count": count, "segmentation_loss": total_loss/count,
            "image_auroc": image["auroc"], "image_ap": image["ap"],
            "pixel_auroc": pixel["auroc"], "pixel_ap": pixel["ap"],
            "pixel_metric_bins": bins,
            "spatial_mode": loader.dataset.data.mode}
