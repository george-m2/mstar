"""Adversarial attack wrappers: FGSM, PGD (L-inf), CW (L-2), AutoAttack (L-inf)
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
    # With set_normalization_used, torchattacks expects NORMALIZED inputs: it
    # inverse-normalizes internally, crafts in [0,1] pixel space (normalizing
    # again before each model call), and returns normalized adversarials. Do
    # not un/re-normalize around the call -- doing both transforms here as
    # well doubles them up, which shifts the point the attack is crafted at
    # and delivers pixel perturbations of ~eps/std (~4.5x the stated budget).
    atk = atk_ctor()
    atk.set_normalization_used(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
    return atk(images, labels)

@dataclass(frozen=True) # what on earth is a frozen dataclass?
class AttackSpec:
    name: str                # "fgsm" | "pgd" | "cw" | "autoattack"
    epsilon: float           # L-inf budget for FGSM/PGD/AutoAttack; unused by CW
    steps: int = 20          # PGD / CW iterations
    alpha: float | None = None  # PGD step size in pixel space; None -> 2.5*eps/steps
    cw_c: float = 1.0        # CW confidence / loss balance
    cw_kappa: float = 0.0
    cw_lr: float = 0.01
    random_start: bool = True
    n_classes: int | None = None  # required by AutoAttack's targeted components
    aa_version: str = "standard"  # "standard" | "rand"


def resolve_pgd_alpha(epsilon: float, steps: int, alpha: float | None) -> float:
    # 2.5*eps/steps (Madry et al.) -- a fixed alpha equal to eps degenerates
    # PGD towards a randomly-restarted FGSM at small budgets.
    if alpha is not None:
        return alpha
    return 2.5 * epsilon / max(steps, 1)


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
                alpha=resolve_pgd_alpha(spec.epsilon, spec.steps, spec.alpha),
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
    elif name == "autoattack":
        # Croce & Hein (2020): APGD-CE + APGD-T + FAB-T + Square, parameter-free.
        # torchattacks ports the reference implementation. The targeted
        # components need the class count; Square gives a query-based
        # black-box attack for free.
        if spec.n_classes is None:
            raise ValueError("AutoAttack requires AttackSpec.n_classes.")
        def ctor():
            return _ta.AutoAttack(
                model,
                norm="Linf",
                eps=spec.epsilon,
                version=spec.aa_version,
                n_classes=spec.n_classes,
                seed=0,
            )
    else:
        raise ValueError(f"Unknown attack '{spec.name}'.")

    def fn(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return _run_torchattack(ctor, model, images, labels)
    return fn


def pgd_linf(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """L-inf PGD on normalized inputs, crafted in pixel space.

    Lightweight inline implementation for the adversarial-training inner loop,
    where constructing a torchattacks object per batch is wasteful. Gradients
    are taken in full precision regardless of AMP settings.
    """
    pixel = _normalize_to_pixel(images).detach()
    if random_start:
        adv = (pixel + torch.empty_like(pixel).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    else:
        adv = pixel.clone()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(_pixel_to_normalize(adv)), labels)
        grad = torch.autograd.grad(loss, adv)[0]
        with torch.no_grad():
            adv = adv + alpha * grad.sign()
            adv = pixel + (adv - pixel).clamp(-epsilon, epsilon)
            adv = adv.clamp(0.0, 1.0)
        adv = adv.detach()
    return _pixel_to_normalize(adv)


@torch.no_grad()
def perturbation_norms(
    clean: torch.Tensor, adv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example (L2, L-inf) perturbation norms in pixel space.

    CW is a minimum-distortion attack: the meaningful statistic is the
    distortion it needed, not accuracy at a fixed budget, so evaluation
    records these norms alongside accuracy.
    """
    delta = _normalize_to_pixel(adv) - _normalize_to_pixel(clean)
    flat = delta.flatten(start_dim=1)
    return flat.norm(p=2, dim=1), flat.abs().max(dim=1).values
