"""Shared constants and path resolution for training / attack pipelines."""

from __future__ import annotations

import os
from pathlib import Path


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

SUPPORTED_MODELS: tuple[str, ...] = ("resnet50", "efficientnet_b3", "vit_b_16")
SUPPORTED_ATTACKS: tuple[str, ...] = ("fgsm", "pgd", "cw")
SUPPORTED_DATASETS: tuple[str, ...] = ("mstar", "atrnet_star")


def project_root() -> Path:
    return Path(os.getenv("SAR_ATR_PROJECT_DIR", Path.cwd())).resolve()


def default_checkpoint_dir() -> Path:
    return Path(os.getenv("SAR_ATR_CHECKPOINT_DIR", project_root() / "checkpoints")).resolve()


def default_results_dir() -> Path:
    return Path(os.getenv("SAR_ATR_RESULTS_DIR", project_root() / "results")).resolve()


def default_model_cache_dir() -> Path:
    return Path(os.getenv("SAR_ATR_MODEL_CACHE_DIR", project_root() / ".torch")).resolve()


def run_dir(dataset: str, model: str, seed: int, root: Path | None = None) -> Path:
    """Directory for a single (dataset, model, seed) training run."""
    base = root or default_checkpoint_dir()
    return base / dataset / model / f"seed_{seed}"
