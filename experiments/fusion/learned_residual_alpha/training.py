"""Optimization loop for the learned residual-alpha fusion backend."""

import numpy as np
import torch
from tqdm import tqdm


def train_backend(
    backend,
    sample_dataloader,
    *,
    epochs: int | None = None,
    lr: float | None = None,
    checkpoint_path: str | None = None,
    device=None,
    config=None,
) -> dict:
    """Train ``backend`` while keeping orchestration outside its runtime class."""
    epochs = int(epochs if epochs is not None else backend.params.train_epochs)
    lr = float(lr if lr is not None else backend.params.train_lr)
    backend.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = None
    history: list[float] = []
    max_samples = backend.params.train_max_samples_per_epoch
    log_every = backend.params.log_every

    for epoch in range(epochs):
        losses = []
        iterator = tqdm(sample_dataloader, desc=f"Fusion backend epoch {epoch + 1}/{epochs}")
        for sample_idx, sample in enumerate(iterator):
            if max_samples is not None and sample_idx >= int(max_samples):
                break

            prepared = backend._prepare_training_sample(sample)
            if prepared is None:
                continue

            features, target, control, anomaly, base_alpha, support, mask, scale = prepared
            if backend.model is None:
                backend.warmup(tuple(target.shape[1:]), device=backend.device)
            if optimizer is None:
                optimizer = torch.optim.AdamW(
                    backend.model.parameters(),
                    lr=lr,
                    weight_decay=float(backend.params.train_weight_decay),
                )
                backend.model.train()

            optimizer.zero_grad(set_to_none=True)
            fused, alpha_delta, residual = backend._forward_components(
                features,
                control,
                anomaly,
                base_alpha,
                support,
                scale,
            )
            loss = backend._training_loss(fused, target, alpha_delta, residual, mask, support)
            loss.backward()
            grad_clip_norm = backend.params.grad_clip_norm
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(backend.model.parameters(), float(grad_clip_norm))
            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            if log_every and (sample_idx + 1) % int(log_every) == 0:
                iterator.set_postfix(loss=f"{np.mean(losses[-int(log_every):]):.5f}")

        if not losses:
            raise ValueError("No valid anomalous samples were found for fusion backend training.")
        history.append(float(np.mean(losses)))

    if checkpoint_path is not None:
        backend.save_checkpoint(checkpoint_path, train_loss_history=history)

    backend.model.eval()
    return {"train_loss_history": history, "checkpoint_path": checkpoint_path}


__all__ = ["train_backend"]
