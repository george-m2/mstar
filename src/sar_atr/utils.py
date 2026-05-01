"""Miscellaneous helpers: seeding, device selection, JSON/CSV logging."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (including CUDA) for reproducibility.

    We intentionally do NOT toggle `torch.backends.cudnn.deterministic = True`
    globally -- that can halve training throughput on L40. Set the env var
    `SAR_ATR_DETERMINISTIC=1` to enable it when bit-exact reproducibility is
    required (e.g. for debugging).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if os.getenv("SAR_ATR_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS support kept for local Mac dev; cluster will always hit the CUDA branch.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append `row` to `path`, writing a header if the file is new.

    Safe for repeated calls from different SLURM array tasks because we open
    in append mode and write the header only when the file does not exist.
    Concurrent writes from many tasks may still interleave; the SLURM scripts
    use per-task CSVs that are aggregated after the array completes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def cuda_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "cuda: unavailable"
    props = torch.cuda.get_device_properties(0)
    alloc_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    total_gb = props.total_memory / 1e9
    return (
        f"cuda: {props.name} | total={total_gb:.1f}GB "
        f"allocated={alloc_gb:.1f}GB reserved={reserved_gb:.1f}GB"
    )
