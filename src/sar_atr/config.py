"""Shared constants and path resolution (for use in HPC) for both pipelines"""

from __future__ import annotations

import os
from pathlib import Path


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

SUPPORTED_MODELS: tuple[str, ...] = ("resnet50", "efficientnet_b3", "vit_b_16", "aconvnet")
SUPPORTED_ATTACKS: tuple[str, ...] = ("fgsm", "pgd", "cw", "autoattack")
SUPPORTED_DATASETS: tuple[str, ...] = ("mstar", "atrnet_star")

# Per-architecture fine-tuning recipes. The uniform AdamW lr=1e-3 recipe that
# worked for the CNNs destroys ViT-B/16's pretrained features (74.7% test acc,
# train acc 99.9% (catastrophic forgetting followed by memorisation), so ViT
# gets the standard transformer fine-tuning treatment: low LR, warmup,
# layer-wise LR decay, and label smoothing 
TRAIN_RECIPES: dict[str, dict] = {
    "resnet50": {
        "lr": 1e-3, "weight_decay": 1e-4, "label_smoothing": 0.0,
        "warmup_epochs": 0, "layer_decay": None,
    },
    "efficientnet_b3": {
        "lr": 1e-3, "weight_decay": 1e-4, "label_smoothing": 0.0,
        "warmup_epochs": 0, "layer_decay": None,
    },
    "vit_b_16": {
        "lr": 1e-4, "weight_decay": 0.05, "label_smoothing": 0.1,
        "warmup_epochs": 5, "layer_decay": 0.75,
    },
    # SAR-native baseline (Chen et al. 2016), trained from scratch -- answers
    # the "vulnerability is an artifact of ImageNet transfer" objection.
    "aconvnet": {
        "lr": 1e-3, "weight_decay": 4e-3, "label_smoothing": 0.0,
        "warmup_epochs": 0, "layer_decay": None,
    },
}


def project_root() -> Path:
    return Path(os.getenv("SAR_ATR_PROJECT_DIR", Path.cwd())).resolve()
def default_checkpoint_dir() -> Path:
    return Path(os.getenv("SAR_ATR_CHECKPOINT_DIR", project_root() / "checkpoints")).resolve()
def default_results_dir() -> Path:
    return Path(os.getenv("SAR_ATR_RESULTS_DIR", project_root() / "results")).resolve()
def default_model_cache_dir() -> Path:
    return Path(os.getenv("SAR_ATR_MODEL_CACHE_DIR", project_root() / ".torch")).resolve()
def run_dir(dataset: str, model: str, seed: int, root: Path | None = None) -> Path:
    base = root or default_checkpoint_dir()
    return base / dataset / model / f"seed_{seed}"
