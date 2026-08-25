from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .config import SUPPORTED_MODELS


class AConvNet(nn.Module):
    """All-convolutional SAR-ATR network of Chen et al. (2016).
    """

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=6), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5), nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=3)

    def forward(self, x):
        logits = self.classifier(self.features(x))
        return logits.mean(dim=(2, 3))


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


def _build_aconvnet(num_classes: int, pretrained: bool) -> nn.Module:
    return AConvNet(num_classes)


_BUILDERS = {
    "resnet50": _build_resnet50,
    "efficientnet_b3": _build_efficientnet_b3,
    "vit_b_16": _build_vit_b_16,
    "aconvnet": _build_aconvnet,
}

def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. Choose from {SUPPORTED_MODELS}."
        )
    return _BUILDERS[name](num_classes, pretrained)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _vit_layer_id(param_name: str, num_layers: int) -> int:
    # 0 = patch/positional embeddings, 1..num_layers = encoder blocks,
    # num_layers + 1 = final LayerNorm and classifier head.
    if param_name.startswith(("conv_proj", "class_token", "encoder.pos_embedding")):
        return 0
    if param_name.startswith("encoder.layers.encoder_layer_"):
        idx = int(param_name.removeprefix("encoder.layers.encoder_layer_").split(".")[0])
        return idx + 1
    return num_layers + 1


def build_param_groups(
    model: nn.Module,
    model_name: str,
    base_lr: float,
    weight_decay: float,
    layer_decay: float | None = None,
) -> list[dict]:
    if layer_decay is None:
        return [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]

    if model_name != "vit_b_16":
        raise ValueError(f"layer_decay is only implemented for vit_b_16, got '{model_name}'.")

    num_layers = len(model.encoder.layers)
    max_id = num_layers + 1
    groups: dict[tuple[int, bool], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _vit_layer_id(name, num_layers)
        no_decay = param.ndim <= 1 or name in ("class_token", "encoder.pos_embedding")
        key = (layer_id, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * (layer_decay ** (max_id - layer_id)),
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(param)
    return list(groups.values())