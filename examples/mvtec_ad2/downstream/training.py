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

from .draem.model import DRAEM
from .draem.losses import training_loss
from .evaluation import evaluate, evaluation_loader


def train(dataset, validation, config, output, *, initial_checkpoint=None):
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
    model = DRAEM(settings.reconstruction_width, settings.segmentation_width).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sorted(set([max(1, int(settings.epochs*.8)), max(1, int(settings.epochs*.9))])), gamma=.2)
    loader = DataLoader(dataset, batch_size=settings.batch_size, num_workers=settings.num_workers,
                        generator=torch.Generator().manual_seed(config.seed))
    validation_loader = evaluation_loader(validation, settings)
    output = Path(output)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    with (output / "training_history.csv").open("w", newline="") as file:
        fields = ["epoch", "train_loss", "validation_loss", "normal", "draem", "hybrid"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(settings.epochs):
            dataset.set_epoch(epoch)
            model.train()
            total, seen = 0., 0
            counts = Counter()
            for batch in tqdm(loader, desc=f"DRAEM epoch {epoch+1}/{settings.epochs}"):
                image, target, mask = (batch[key].to(device) for key in ("image", "target", "mask"))
                optimizer.zero_grad(set_to_none=True)
                reconstruction, logits = model(image)
                loss = training_loss(reconstruction, logits, target, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite DRAEM training loss.")
                loss.backward()
                optimizer.step()
                total += loss.item()*len(image)
                seen += len(image)
                counts.update(batch["source"])
            metrics = evaluate(model, validation_loader, device, patch_batch_size=settings.batch_size)
            score = metrics["segmentation_loss"]
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite DRAEM validation loss.")
            scheduler.step()
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "epoch": epoch+1,
                          "configuration": config.to_dict(), "validation": metrics}
            torch.save(checkpoint, checkpoints / "last.pt")
            if score < best:
                best = score
                torch.save(checkpoint, checkpoints / "best.pt")
            writer.writerow({"epoch": epoch+1, "train_loss": total/seen, "validation_loss": score,
                             **{source: counts[source] for source in ("normal", "draem", "hybrid")}})
            file.flush()
    return output
