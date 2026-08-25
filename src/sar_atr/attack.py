"""
Adverserial attack of a given SAR-ATR checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from sar_atr.attacks import AttackSpec, build_attack, resolve_pgd_alpha
from sar_atr.config import (
    SUPPORTED_ATTACKS, SUPPORTED_DATASETS, SUPPORTED_MODELS,
    default_model_cache_dir, default_results_dir, run_dir,
)
from sar_atr.datasets import load_dataset, stratified_subset_loader
from sar_atr.engine import evaluate, evaluate_adversarial
from sar_atr.models import build_model
from sar_atr.utils import (
    append_csv_row, cuda_memory_summary, cuda_peak_memory_gb, get_logger,
    reset_cuda_peak_stats, save_json, seed_everything, select_device,
)

# all arguments are required except where noted
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adversarial evaluation of a SAR ATR checkpoint.",
    )
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    p.add_argument("--dataset", default="atrnet_star", choices=SUPPORTED_DATASETS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--attack_type", required=True, choices=SUPPORTED_ATTACKS)
    p.add_argument("--epsilon", type=float, required=True,
                   help="L-inf budget in pixel space (ignored by CW, which uses --cw_c).")
    p.add_argument("--checkpoint_path", type=Path, default=None,
                   help="Weights to attack. Defaults to "
                        "checkpoints/{dataset}/{model}/seed_{seed}/best_model.pth.")
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--atrnet_config", default="SOC-40")
    # Attack hyperparameters
    p.add_argument("--pgd_steps", type=int, default=20)
    p.add_argument("--pgd_alpha", type=float, default=None,
                   help="PGD step size; defaults to 2.5*eps/steps.")
    p.add_argument("--cw_steps", type=int, default=50)
    p.add_argument("--cw_c", type=float, default=1.0)
    p.add_argument("--cw_kappa", type=float, default=0.0)
    p.add_argument("--cw_lr", type=float, default=0.01)
    p.add_argument("--aa_version", default="standard", choices=("standard", "rand"),
                   help="AutoAttack version.")
    p.add_argument("--subset_size", type=int, default=None,
                   help="Evaluate on a fixed class-stratified test subset. "
                        "Recommended 5000 for AutoAttack; unset = full test set.")
    # Outputs
    p.add_argument("--results_csv", type=Path, default=None,
                   help="Append a summary row here. Defaults to results/attack_results.csv.")
    p.add_argument("--results_json", type=Path, default=None,
                   help="Optional per-run JSON dump (auto-named under output_dir if unset).")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where to write per-run logs / JSON. Defaults to run_dir/attacks/{attack}_eps{eps}.")
    p.add_argument("--skip_clean_eval", action="store_true",
                   help="Skip clean-accuracy measurement (useful if another task already did it).")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    run_root = run_dir(args.dataset, args.model, args.seed)
    ckpt = args.checkpoint_path or (run_root / "best_model.pth")
    out_dir = args.output_dir or (
        run_root / "attacks" / f"{args.attack_type}_eps{args.epsilon:.4f}"
    )
    csv_path = args.results_csv or (default_results_dir() / "attack_results.csv")
    json_path = args.results_json or (out_dir / "result.json")
    return ckpt, out_dir, csv_path, json_path


def main() -> int:
    args = parse_args()
    ckpt_path, out_dir, csv_path, json_path = _resolve_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(
        f"attack.{args.model}.s{args.seed}.{args.attack_type}.e{args.epsilon:.3f}",
        log_file=out_dir / "attack.log",
    )
    logger.info("args = %s", json.dumps(vars(args), default=str))

    if not ckpt_path.exists():
        logger.error("checkpoint not found: %s", ckpt_path)
        return 2

    seed_everything(args.seed)
    device = select_device()
    reset_cuda_peak_stats()
    logger.info("device=%s | %s", device, cuda_memory_summary())

    os.environ.setdefault("TORCH_HOME", str(default_model_cache_dir()))
    torch.hub.set_dir(str(default_model_cache_dir()))

    data = load_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        num_workers=args.num_workers,
        image_size=args.image_size,
        atrnet_config=args.atrnet_config,
    )
    test_loader = data.test
    if args.subset_size is not None:
        # Fixed function of (dataset, size, seed=0): every model/attack pair
        # sees the same images, so subset numbers are comparable across runs.
        test_loader = stratified_subset_loader(test_loader, args.subset_size, seed=0)
    logger.info(
        "dataset=%s test=%d (of %d) classes=%d",
        args.dataset, len(test_loader.dataset), len(data.test.dataset),
        data.num_classes,
    )

    model = build_model(
        args.model, num_classes=data.num_classes, pretrained=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state if not isinstance(state, dict) or "model_state_dict" not in state
                         else state["model_state_dict"])
    model.eval()
    logger.info("loaded weights from %s", ckpt_path)

    criterion = nn.CrossEntropyLoss()

    clean_acc: float | None = None
    if not args.skip_clean_eval:
        clean_res = evaluate(model, test_loader, criterion, device, desc="clean")
        clean_acc = float(clean_res.accuracy)
        logger.info("clean_acc=%.4f", clean_acc)

    pgd_alpha = resolve_pgd_alpha(args.epsilon, args.pgd_steps, args.pgd_alpha)
    spec = AttackSpec(
        name=args.attack_type,
        epsilon=args.epsilon,
        steps=args.cw_steps if args.attack_type == "cw" else args.pgd_steps,
        alpha=pgd_alpha,
        cw_c=args.cw_c,
        cw_kappa=args.cw_kappa,
        cw_lr=args.cw_lr,
        n_classes=data.num_classes,
        aa_version=args.aa_version,
    )
    attack_fn = build_attack(spec, model)

    t0 = time.time()
    adv_acc, pert_stats = evaluate_adversarial(
        model, test_loader, attack_fn, device,
        desc=f"{args.attack_type} eps={args.epsilon}",
        return_perturbation_stats=True,
    )
    attack_sec = time.time() - t0

    attack_success_rate = 1.0 - adv_acc
    peak_gb = cuda_peak_memory_gb()

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "attack_type": args.attack_type,
        "epsilon": float(args.epsilon),
        "pgd_steps": int(args.pgd_steps) if args.attack_type == "pgd" else None,
        "pgd_alpha": float(pgd_alpha) if args.attack_type == "pgd" else None,
        "cw_c": float(args.cw_c) if args.attack_type == "cw" else None,
        "cw_steps": int(args.cw_steps) if args.attack_type == "cw" else None,
        "aa_version": args.aa_version if args.attack_type == "autoattack" else None,
        "n_eval": len(test_loader.dataset),
        "subset_size": args.subset_size,
        "clean_acc": clean_acc,
        "adv_acc": float(adv_acc),
        "attack_success_rate": float(attack_success_rate),
        # Pixel-space perturbation norms over successful attacks. For CW the
        # median L2 is the reportable statistic; for L-inf attacks linf_max
        # confirms the budget was respected.
        "pert_n_success": pert_stats["n_success"],
        "pert_l2_median": pert_stats["l2_median"],
        "pert_l2_mean": pert_stats["l2_mean"],
        "pert_linf_median": pert_stats["linf_median"],
        "pert_linf_max": pert_stats["linf_max"],
        "attack_sec": float(attack_sec),
        "peak_vram_gb": peak_gb,
        "checkpoint": str(ckpt_path),
    }
    save_json(json_path, result)
    logger.info(
        "attack=%s eps=%.4f | clean_acc=%s adv_acc=%.4f asr=%.4f "
        "l2_median=%s time=%.1fs peak_vram=%s",
        args.attack_type, args.epsilon,
        f"{clean_acc:.4f}" if clean_acc is not None else "skipped",
        adv_acc, attack_success_rate,
        f"{pert_stats['l2_median']:.3f}" if pert_stats["l2_median"] is not None else "n/a",
        attack_sec,
        f"{peak_gb:.1f}GB" if peak_gb is not None else "n/a",
    )
    append_csv_row(csv_path, result)
    return 0