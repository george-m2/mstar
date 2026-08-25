"""Black-box transferability evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from sar_atr.attacks import AttackSpec, build_attack
from sar_atr.config import (
    SUPPORTED_DATASETS, SUPPORTED_MODELS,
    default_model_cache_dir, default_results_dir, run_dir,
)
from sar_atr.datasets import load_dataset, stratified_subset_loader
from sar_atr.models import build_model
from sar_atr.utils import (
    append_csv_row, cuda_memory_summary, cuda_peak_memory_gb, get_logger,
    reset_cuda_peak_stats, save_json, seed_everything, select_device,
)


def parse_model_spec(spec: str) -> tuple[str, int, Path | None]:
    """Parse "model:seed" or "model:seed:/path/to/best_model.pth"."""
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"Expected model:seed[:checkpoint], got '{spec}'."
        )
    model, seed = parts[0], int(parts[1])
    if model not in SUPPORTED_MODELS:
        raise argparse.ArgumentTypeError(
            f"Unknown model '{model}'. Choose from {SUPPORTED_MODELS}."
        )
    ckpt = Path(parts[2]) if len(parts) == 3 else None
    return model, seed, ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transferability of adversarial examples across checkpoints.",
    )
    p.add_argument("--dataset", default="atrnet_star", choices=SUPPORTED_DATASETS)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--source", type=parse_model_spec, required=True,
                   help="model:seed[:checkpoint] to craft the attack on.")
    p.add_argument("--targets", type=parse_model_spec, nargs="+", required=True,
                   help="One or more model:seed[:checkpoint] to evaluate on.")
    p.add_argument("--attack_type", default="pgd", choices=("fgsm", "pgd"))
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--pgd_steps", type=int, default=20)
    p.add_argument("--pgd_alpha", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--atrnet_config", default="SOC-40")
    p.add_argument("--subset_size", type=int, default=None,
                   help="Optional fixed stratified test subset.")
    p.add_argument("--results_csv", type=Path, default=None,
                   help="Defaults to results/transfer_results.csv.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where logs/JSON go; defaults under the source run dir.")
    return p.parse_args()


def _load_checkpoint(model_name: str, seed: int, ckpt: Path | None,
                     dataset: str, num_classes: int, device) -> nn.Module:
    path = ckpt or (run_dir(dataset, model_name, seed) / "best_model.pth")
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(
        state if not isinstance(state, dict) or "model_state_dict" not in state
        else state["model_state_dict"]
    )
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    src_model_name, src_seed, src_ckpt = args.source

    out_dir: Path = args.output_dir or (
        run_dir(args.dataset, src_model_name, src_seed)
        / "transfer" / f"{args.attack_type}_eps{args.epsilon:.4f}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_csv or (default_results_dir() / "transfer_results.csv")

    logger = get_logger(
        f"transfer.{src_model_name}.s{src_seed}.{args.attack_type}.e{args.epsilon:.3f}",
        log_file=out_dir / "transfer.log",
    )
    logger.info("args = %s", json.dumps(vars(args), default=str))

    seed_everything(src_seed)
    device = select_device()
    reset_cuda_peak_stats()
    logger.info("device=%s | %s", device, cuda_memory_summary())

    os.environ.setdefault("TORCH_HOME", str(default_model_cache_dir()))
    torch.hub.set_dir(str(default_model_cache_dir()))

    data = load_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=src_seed,
        device=device,
        num_workers=args.num_workers,
        image_size=args.image_size,
        atrnet_config=args.atrnet_config,
    )
    test_loader = data.test
    if args.subset_size is not None:
        test_loader = stratified_subset_loader(test_loader, args.subset_size, seed=0)
    logger.info("test=%d classes=%d", len(test_loader.dataset), data.num_classes)

    source = _load_checkpoint(
        src_model_name, src_seed, src_ckpt, args.dataset, data.num_classes, device,
    )
    targets = [
        (name, seed, _load_checkpoint(name, seed, ckpt, args.dataset,
                                      data.num_classes, device))
        for name, seed, ckpt in args.targets
    ]
    logger.info(
        "source=%s:s%d | targets=%s",
        src_model_name, src_seed,
        ", ".join(f"{n}:s{s}" for n, s, _ in targets),
    )

    spec = AttackSpec(
        name=args.attack_type,
        epsilon=args.epsilon,
        steps=args.pgd_steps,
        alpha=args.pgd_alpha,
    )
    attack_fn = build_attack(spec, source)

    # Counters: source clean/adv, and per-target clean/adv plus the joint
    # counts needed for the conditional transfer rate.
    n_total = 0
    src_clean_correct = 0
    src_adv_correct = 0
    t_stats = [
        {"clean_correct": 0, "adv_correct": 0, "eligible": 0, "transferred": 0}
        for _ in targets
    ]

    t0 = time.time()
    for images, labels in tqdm(test_loader, desc="transfer", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv = attack_fn(images, labels)

        with torch.no_grad():
            src_clean_ok = source(images).argmax(1).eq(labels)
            src_adv_ok = source(adv).argmax(1).eq(labels)
            src_fooled = ~src_adv_ok

            n_total += labels.size(0)
            src_clean_correct += int(src_clean_ok.sum())
            src_adv_correct += int(src_adv_ok.sum())

            for stats, (_, _, target) in zip(t_stats, targets):
                tgt_clean_ok = target(images).argmax(1).eq(labels)
                tgt_adv_ok = target(adv).argmax(1).eq(labels)
                stats["clean_correct"] += int(tgt_clean_ok.sum())
                stats["adv_correct"] += int(tgt_adv_ok.sum())
                eligible = src_fooled & tgt_clean_ok
                stats["eligible"] += int(eligible.sum())
                stats["transferred"] += int((eligible & ~tgt_adv_ok).sum())
    elapsed = time.time() - t0

    src_clean_acc = src_clean_correct / max(n_total, 1)
    src_adv_acc = src_adv_correct / max(n_total, 1)
    peak_gb = cuda_peak_memory_gb()
    logger.info(
        "source %s:s%d | clean_acc=%.4f adv_acc=%.4f (white-box sanity)",
        src_model_name, src_seed, src_clean_acc, src_adv_acc,
    )

    rows = []
    for stats, (tgt_name, tgt_seed, _) in zip(t_stats, targets):
        row = {
            "dataset": args.dataset,
            "attack_type": args.attack_type,
            "epsilon": float(args.epsilon),
            "pgd_steps": int(args.pgd_steps) if args.attack_type == "pgd" else None,
            "source_model": src_model_name,
            "source_seed": src_seed,
            "target_model": tgt_name,
            "target_seed": tgt_seed,
            "n_eval": n_total,
            "subset_size": args.subset_size,
            "source_clean_acc": src_clean_acc,
            "source_adv_acc": src_adv_acc,
            "target_clean_acc": stats["clean_correct"] / max(n_total, 1),
            "target_adv_acc": stats["adv_correct"] / max(n_total, 1),
            "n_transfer_eligible": stats["eligible"],
            "transfer_rate": (
                stats["transferred"] / stats["eligible"] if stats["eligible"] else None
            ),
            "elapsed_sec": float(elapsed),
        }
        rows.append(row)
        append_csv_row(csv_path, row)
        logger.info(
            "target %s:s%d | clean=%.4f adv=%.4f transfer_rate=%s (n=%d)",
            tgt_name, tgt_seed,
            row["target_clean_acc"], row["target_adv_acc"],
            f"{row['transfer_rate']:.4f}" if row["transfer_rate"] is not None else "n/a",
            stats["eligible"],
        )

    save_json(out_dir / "result.json", {"rows": rows, "peak_vram_gb": peak_gb})
    logger.info("done | %d targets in %.1fs peak_vram=%s",
                len(targets), elapsed,
                f"{peak_gb:.1f}GB" if peak_gb is not None else "n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
