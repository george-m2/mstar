from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .config import SUPPORTED_MODELS


def _build_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _build_efficientnet_b3(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b3(weights=weights)
    # torchvision EfficientNet exposes `classifier` as Sequential(Dropout, Linear).
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def _build_vit_b_16(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    m = models.vit_b_16(weights=weights)
    in_features = m.heads.head.in_features
    m.heads.head = nn.Linear(in_features, num_classes)
    return m


_BUILDERS = {
    "resnet50": _build_resnet50,
    "efficientnet_b3": _build_efficientnet_b3,
    "vit_b_16": _build_vit_b_16,
}

def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. Choose from {SUPPORTED_MODELS}."
        )
    return _BUILDERS[name](num_classes, pretrained)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
