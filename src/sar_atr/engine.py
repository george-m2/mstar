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

    # `grad_accum_steps > 1` lets ViT-B fit into L40 VRAM by splitting each 
    # effective batch across multiple forward/backward passes before stepping.
    
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    amp_enabled = use_amp and device.type == "cuda"
    # `torch.amp.GradScaler("cuda", ...)` is the torch 2.4+ API; fall back to
    # the legacy namespace for older torches so this file stays portable.
    # no idea what API version ill have...
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

    # training state MUST match the loss.
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
    return_perturbation_stats: bool = False,
) -> float | tuple[float, dict]:
    """Adversarial accuracy; optionally also perturbation-norm statistics.

    With `return_perturbation_stats=True`, returns (accuracy, stats) where
    stats holds pixel-space L2/L-inf norms of successful perturbations. For
    minimum-distortion attacks (CW) the median L2 is the headline number; for
    budget-constrained attacks it is a sanity check that the budget held.
    """
    from .attacks import perturbation_norms  # local import to avoid a cycle

    model.eval()
    correct, total = 0, 0
    success_l2: list[torch.Tensor] = []
    success_linf: list[torch.Tensor] = []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv = attack_fn(images, labels)
        with torch.no_grad():
            outputs = model(adv)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            if return_perturbation_stats:
                l2, linf = perturbation_norms(images, adv)
                fooled = ~predicted.eq(labels)
                success_l2.append(l2[fooled].cpu())
                success_linf.append(linf[fooled].cpu())

    accuracy = correct / max(total, 1)
    if not return_perturbation_stats:
        return accuracy

    if success_l2 and (all_l2 := torch.cat(success_l2)).numel() > 0:
        all_linf = torch.cat(success_linf)
        stats = {
            "n_success": int(all_l2.numel()),
            "l2_median": float(all_l2.median()),
            "l2_mean": float(all_l2.mean()),
            "linf_median": float(all_linf.median()),
            "linf_max": float(all_linf.max()),
        }
    else:
        stats = {
            "n_success": 0, "l2_median": None, "l2_mean": None,
            "linf_median": None, "linf_max": None,
        }
    return accuracy, stats


def train_one_epoch_adversarial(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attack_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    use_amp: bool = True,
    desc: str = "AT train",
) -> EpochResult:
    """One epoch of Madry-style adversarial training.

    The model trains purely on adversarial examples. Crafting runs with the
    model in eval mode so BatchNorm batch statistics are frozen and running
    stats are not updated k extra times per batch; the update step runs in
    train mode as usual. Reported accuracy is on the adversarial examples.
    """
    total_loss, correct, total = 0.0, 0, 0
    amp_enabled = use_amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)  # type: ignore[attr-defined]

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.eval()
        adv = attack_fn(images, labels)
        model.train()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=amp_enabled,
        ):
            outputs = model(adv)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return EpochResult(loss=total_loss / max(total, 1), accuracy=correct / max(total, 1))

# CHECKPOINTING

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
