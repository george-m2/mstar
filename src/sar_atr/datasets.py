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


    eval_ds = datasets.ImageFolder(str(data_dir), transform=test_tf)
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

    root = Path(data_dir)
    candidates = [root / experimental_config, root]
    cfg_root = next((c for c in candidates if (c / "train").is_dir() and (c / "test").is_dir()), None)
    if cfg_root is None:
        raise FileNotFoundError(
        )

    train_tf, test_tf = build_transforms(image_size=image_size, augment=True)

    train_full = datasets.ImageFolder(
        str(cfg_root / "train"), transform=train_tf, loader=_rgb_loader,
    )
    test_ds = datasets.ImageFolder(
        str(cfg_root / "test"), transform=test_tf, loader=_rgb_loader,
    )

    # seed-dependent slice out of the train split.
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


def _dataset_targets(ds) -> list[int]:
    if isinstance(ds, Subset):
        parent = _dataset_targets(ds.dataset)
        return [parent[i] for i in ds.indices]
    return list(ds.targets)


def stratified_subset_loader(
    loader: DataLoader,
    size: int,
    seed: int = 0,
) -> DataLoader:
    """Deterministic class-stratified subset of an evaluation DataLoader.
    """
    ds = loader.dataset
    targets = _dataset_targets(ds)
    n_total = len(targets)
    if size >= n_total:
        return loader

    by_class: dict[int, list[int]] = {}
    for idx, t in enumerate(targets):
        by_class.setdefault(int(t), []).append(idx)

    gen = torch.Generator().manual_seed(seed)
    chosen: list[int] = []
    # Proportional allocation with at least one example per class.
    for cls in sorted(by_class):
        idxs = by_class[cls]
        take = max(1, round(size * len(idxs) / n_total))
        perm = torch.randperm(len(idxs), generator=gen).tolist()
        chosen.extend(idxs[i] for i in perm[:take])
    chosen = sorted(chosen[:size] if len(chosen) > size else chosen)

    return DataLoader(
        Subset(ds, chosen),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
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
