"""Adversarial attack wrappers: FGSM, PGD (L-inf), CW (L-2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks as _ta

from .config import IMAGENET_MEAN, IMAGENET_STD

def _normalize_to_pixel(images: torch.Tensor) -> torch.Tensor:
    # Un-normalize ImageNet-normalized tensor back to [0, 1]
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp_(0.0, 1.0)


def _pixel_to_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _run_torchattack(
    atk_ctor: Callable[[], "object"],
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    pixel_images = _normalize_to_pixel(images)
    atk = atk_ctor()
    # Tell torchattacks that downstream model inference happens on normalized
    # inputs, so it normalizes internally when computing logits.
    atk.set_normalization_used(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
    adv_pixels = atk(pixel_images, labels)
    return _pixel_to_normalize(adv_pixels)

@dataclass(frozen=True) # what on earth is a frozen dataclass? 
class AttackSpec:
    name: str                # "fgsm" | "pgd" | "cw"
    epsilon: float           # L-inf budget for FGSM/PGD; interpreted as `c` scale for CW (unused)
    steps: int = 20          # PGD / CW iterations
    alpha: float = 0.01      # PGD step size in pixel space
    cw_c: float = 1.0        # CW confidence / loss balance
    cw_kappa: float = 0.0
    cw_lr: float = 0.01
    random_start: bool = True


def build_attack(spec: AttackSpec, model: nn.Module) -> Callable[
    [torch.Tensor, torch.Tensor], torch.Tensor
]:
    name = spec.name.lower()

    if name == "fgsm":
        def ctor():
            return _ta.FGSM(model, eps=spec.epsilon)
    elif name == "pgd":
        def ctor():
            return _ta.PGD(
                model,
                eps=spec.epsilon,
                alpha=spec.alpha,
                steps=spec.steps,
                random_start=spec.random_start,
            )
        # PGDL2 is available in torchattacks; we stick to L-inf PGD per plan.
    elif name == "cw":
        def ctor():
            return _ta.CW(
                model,
                c=spec.cw_c,
                kappa=spec.cw_kappa,
                steps=spec.steps if spec.steps >= 50 else 50,
                lr=spec.cw_lr,
            )
    else:
        raise ValueError(f"Unknown attack '{spec.name}'.")

    def fn(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return _run_torchattack(ctor, model, images, labels)
    return fn
