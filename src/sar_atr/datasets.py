"""Dataset loaders for MSTAR and ATRNet-STAR.

MSTAR -- single flat directory of class folders (`Padded_imgs/<class>/*.jpg`).
ATRNet-STAR -- pre-split hierarchy `<root>/<experimental_config>/{train,test}/<class>/*.tif`
(see https://github.com/waterdisappear/ATRNet-STAR). We default to the SOC-40
configuration, which is the combined full open-source release.

Both loaders return (train_loader, val_loader, test_loader, class_names).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, SUPPORTED_DATASETS


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]
    num_classes: int


def _rgb_loader(path: str) -> Image.Image:
    # ATRNet-STAR 8-bit amplitude is single-channel TIFF; ImageFolder's default
    # PIL loader opens as "L" for those. We explicitly convert to RGB so the
    # 3-channel ImageNet-pretrained backbones Just Work (R=G=B=amplitude).
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def build_transforms(
    image_size: int = 224,
    augment: bool = True,
) -> tuple[Callable, Callable]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            *(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                ]
                if augment
                else []
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, test_tf


def _num_workers(device: torch.device, requested: int | None) -> int:
    if requested is not None:
        return max(0, requested)
    if device.type != "cuda":
        return 0
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return 8 if vram_gb > 40 else (4 if vram_gb > 20 else 2)


def _make_loaders(
    train_ds,
    val_ds,
    test_ds,
    class_names: list[str],
    batch_size: int,
    device: torch.device,
    num_workers: int | None,
) -> DataLoaders:
    nw = _num_workers(device, num_workers)
    pin = device.type == "cuda"
    return DataLoaders(
        train=DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=pin, persistent_workers=nw > 0,
        ),
        val=DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=pin, persistent_workers=nw > 0,
        ),
        test=DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=pin, persistent_workers=nw > 0,
        ),
        class_names=class_names,
        num_classes=len(class_names),
    )


def load_mstar(
    data_dir: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int | None = None,
    image_size: int = 224,
) -> DataLoaders:
    """MSTAR: 70/15/15 split driven by `seed` for statistical robustness."""
    train_tf, test_tf = build_transforms(image_size=image_size, augment=True)
    full = datasets.ImageFolder(str(data_dir), transform=train_tf)

    n_total = len(full)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)

    # random_split returns Subsets sharing `full`; swap in the deterministic
    # test transform for the eval subsets so there's no augmentation leakage.
    eval_ds = datasets.ImageFolder(str(data_dir), transform=test_tf)
    if eval_ds.classes != full.classes or eval_ds.samples != full.samples:
        raise RuntimeError(
            "eval_ds ImageFolder enumerated a different ordering than full -- "
            "random_split indices would be invalid."
        )
    val_ds = Subset(eval_ds, val_ds.indices)
    test_ds = Subset(eval_ds, test_ds.indices)

    return _make_loaders(
        train_ds, val_ds, test_ds, full.classes, batch_size, device, num_workers,
    )


def load_atrnet_star(
    data_dir: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int | None = None,
    image_size: int = 224,
    val_fraction: float = 0.10,
    experimental_config: str = "SOC-40",
) -> DataLoaders:
    """ATRNet-STAR loader using the dataset's pre-split train/test folders.

    The provided train split is further partitioned into train/val using `seed`
    and `val_fraction`, so val/test are disjoint. `data_dir` may point to
    either the archive root (containing the experimental-config folder) or
    directly at the experimental-config folder itself.
    """
    root = Path(data_dir)
    candidates = [root / experimental_config, root]
    cfg_root = next((c for c in candidates if (c / "train").is_dir() and (c / "test").is_dir()), None)
    if cfg_root is None:
        raise FileNotFoundError(
            f"Could not locate ATRNet-STAR train/test folders under {root}. "
            f"Expected `{experimental_config}/train` + `{experimental_config}/test` "
            f"(or `train`/`test` directly under data_dir). "
            f"Pass the archive root as --data_dir or override with --atrnet_config."
        )

    train_tf, test_tf = build_transforms(image_size=image_size, augment=True)

    train_full = datasets.ImageFolder(
        str(cfg_root / "train"), transform=train_tf, loader=_rgb_loader,
    )
    test_ds = datasets.ImageFolder(
        str(cfg_root / "test"), transform=test_tf, loader=_rgb_loader,
    )

    if train_full.classes != test_ds.classes:
        raise RuntimeError(
            "Train/test class lists disagree for ATRNet-STAR -- dataset integrity problem."
        )

    # Carve a seed-dependent validation slice out of the train split.
    n_val = int(val_fraction * len(train_full))
    n_train = len(train_full) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_sub = random_split(train_full, [n_train, n_val], generator=gen)

    # val inherits augmented transforms from train_full; swap to eval transforms.
    val_eval = datasets.ImageFolder(
        str(cfg_root / "train"), transform=test_tf, loader=_rgb_loader,
    )
    if val_eval.classes != train_full.classes or val_eval.samples != train_full.samples:
        raise RuntimeError(
            "val_eval ImageFolder enumerated a different ordering than train_full -- "
            "random_split indices would be invalid. Disk state changed during loading?"
        )
    val_ds = Subset(val_eval, val_sub.indices)

    return _make_loaders(
        train_ds, val_ds, test_ds, train_full.classes, batch_size, device, num_workers,
    )


def load_dataset(
    dataset: str,
    data_dir: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int | None = None,
    image_size: int = 224,
    atrnet_config: str = "SOC-40",
) -> DataLoaders:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {SUPPORTED_DATASETS}.")
    if dataset == "mstar":
        return load_mstar(data_dir, batch_size, seed, device, num_workers, image_size)
    return load_atrnet_star(
        data_dir, batch_size, seed, device, num_workers, image_size,
        experimental_config=atrnet_config,
    )
