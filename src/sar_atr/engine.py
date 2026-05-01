"""Training and evaluation loops shared by `train.py` and `attack.py`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EpochResult:
    loss: float
    accuracy: float


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    def append(self, train: EpochResult, val: EpochResult) -> None:
        self.train_loss.append(train.loss)
        self.train_acc.append(train.accuracy)
        self.val_loss.append(val.loss)
        self.val_acc.append(val.accuracy)

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    desc: str = "Training",
) -> EpochResult:
    """Single training epoch with optional mixed precision and grad accumulation.

    `grad_accum_steps > 1` lets ViT-B fit into L40 VRAM by splitting each
    effective batch across multiple forward/backward passes before stepping.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    amp_enabled = use_amp and device.type == "cuda"
    # `torch.amp.GradScaler("cuda", ...)` is the torch 2.4+ API; fall back to
    # the legacy namespace for older torches so this file stays portable.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)  # type: ignore[attr-defined]

    optimizer.zero_grad(set_to_none=True)
    step_in_accum = 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=amp_enabled,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels) / max(grad_accum_steps, 1)

        scaler.scale(loss).backward()
        step_in_accum += 1
        if step_in_accum >= grad_accum_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0

        total_loss += loss.item() * images.size(0) * max(grad_accum_steps, 1)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    # Flush any trailing partial accumulation so training state matches the loss.
    if step_in_accum > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return EpochResult(loss=total_loss / max(total, 1), accuracy=correct / max(total, 1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> EpochResult:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return EpochResult(loss=total_loss / max(total, 1), accuracy=correct / max(total, 1))


def evaluate_adversarial(
    model: nn.Module,
    loader: DataLoader,
    attack_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    desc: str = "Attacking",
) -> float:
    """Return accuracy under the given attack."""
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv = attack_fn(images, labels)
        with torch.no_grad():
            outputs = model(adv)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_full_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    best_val_acc: float,
    patience_counter: int,
    history: TrainHistory,
    class_names: list[str],
    extra: dict | None = None,
) -> None:
    """Atomic save: write to temp file, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "patience_counter": patience_counter,
        "history": history.to_dict(),
        "class_names": class_names,
    }
    if extra is not None:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def save_weights_only(path: Path, model: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(path)
