"""Adversarial attack wrappers: FGSM, PGD (L-inf), CW (L-2).

This module prefers the `torchattacks` library (the de-facto standard in
adversarial-robustness benchmarking) and falls back to from-scratch
implementations of FGSM and PGD if the library is unavailable -- which
preserves parity with the MSTAR proof-of-concept notebooks.

Inputs throughout are assumed to be ImageNet-normalized tensors (the same
space our dataloaders deliver). For the fallback path we convert `epsilon`
from pixel space to normalized space per channel using IMAGENET_STD.
For torchattacks we use `set_normalization_used(...)` so the library can
perform its own [0,1] <-> normalized conversion internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IMAGENET_MEAN, IMAGENET_STD


def _imagenet_bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    # valid normalized range corresponding to pixel range [0, 1]
    norm_min = (0.0 - mean) / std
    norm_max = (1.0 - mean) / std
    return std, norm_min, norm_max


# ---------------------------------------------------------------------------
# Fallback implementations (no torchattacks dependency)
# ---------------------------------------------------------------------------


def _fgsm_fallback(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    std, norm_min, norm_max = _imagenet_bounds(images.device)
    adv = images.clone().detach().requires_grad_(True)
    outputs = model(adv)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    eps_norm = epsilon / std
    adv = adv + eps_norm * adv.grad.sign()
    adv = torch.max(torch.min(adv, norm_max), norm_min)
    return adv.detach()


def _pgd_fallback(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = 0.01,
    steps: int = 20,
    random_start: bool = True,
) -> torch.Tensor:
    std, norm_min, norm_max = _imagenet_bounds(images.device)
    eps_norm = epsilon / std
    alpha_norm = alpha / std

    orig = images.clone().detach()
    adv = orig.clone().detach()
    if random_start:
        adv = adv + (torch.empty_like(adv).uniform_(-1.0, 1.0) * eps_norm)
        adv = torch.max(torch.min(adv, norm_max), norm_min)

    for _ in range(steps):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha_norm * grad.sign()
        delta = torch.clamp(adv - orig, min=-eps_norm, max=eps_norm)
        adv = orig + delta
        adv = torch.max(torch.min(adv, norm_max), norm_min).detach()
    return adv


# ---------------------------------------------------------------------------
# torchattacks-backed implementations
# ---------------------------------------------------------------------------

try:
    import torchattacks as _ta  # type: ignore[import-not-found]
    _HAS_TA = True
except Exception:  # pragma: no cover - handled at runtime
    _ta = None
    _HAS_TA = False


def _normalize_to_pixel(images: torch.Tensor) -> torch.Tensor:
    """Un-normalize ImageNet-normalized tensor back to [0, 1] pixel space."""
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
    """Run a torchattacks attack on ImageNet-normalized inputs.

    torchattacks operates in [0, 1] pixel space. We convert before the call
    and re-normalize after, so the returned tensor matches our loaders.
    """
    pixel_images = _normalize_to_pixel(images)
    atk = atk_ctor()
    # Tell torchattacks that downstream model inference happens on normalized
    # inputs, so it normalizes internally when computing logits.
    atk.set_normalization_used(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
    adv_pixels = atk(pixel_images, labels)
    return _pixel_to_normalize(adv_pixels)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
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
    """Return a `fn(images, labels) -> adv_images` operating on normalized tensors.

    This indirection lets callers cache the attack object across batches
    (important for CW where per-call setup is non-trivial).
    """
    name = spec.name.lower()

    if _HAS_TA:
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

    # ----- fallback path -----
    if name == "fgsm":
        def fn(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return _fgsm_fallback(model, images, labels, spec.epsilon)
        return fn
    if name == "pgd":
        def fn(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return _pgd_fallback(
                model, images, labels,
                epsilon=spec.epsilon, alpha=spec.alpha,
                steps=spec.steps, random_start=spec.random_start,
            )
        return fn
    if name == "cw":
        raise RuntimeError(
            "CW attack requires the `torchattacks` package. "
            "Install with `uv add torchattacks` or `pip install torchattacks`."
        )
    raise ValueError(f"Unknown attack '{spec.name}'.")
