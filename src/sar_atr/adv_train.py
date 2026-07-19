"""Madry-style PGD adversarial training (Madry et al., 2018).

The AT model differs from the clean baseline ONLY in the training loss (worst
-case adversarial examples inside an L-inf ball), so the comparison isolates
the robustness-accuracy tradeoff. Optimisation hyperparameters come from the
same per-architecture recipe as clean training.

Defaults follow community convention: eps = 8/255, 7 inner PGD steps,
alpha = 2.5*eps/steps. Model selection and early stopping use ROBUST validation
accuracy -- robust overfitting (Rice, Wong & Madry, 2020) means final-epoch
robustness can be several points below the peak, and clean-accuracy selection
would quietly pick a less robust model.

    sar-atr-adv-train \
        --model resnet50 \
        --seed 0 \
        --data_dir /scratch/$USER/datasets/atrnet_star \
        --summary_csv results/adv_train_summary.csv

Cost: ~(k+1)x a clean epoch; 30 epochs at k=7 fits a 12-18h walltime for
ResNet-50 on one L40. --resume continues from the last epoch if preempted.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from sar_atr.attacks import pgd_linf, resolve_pgd_alpha
from sar_atr.config import (
    SUPPORTED_DATASETS, SUPPORTED_MODELS,
    default_model_cache_dir, run_dir,
)
from sar_atr.datasets import load_dataset
from sar_atr.engine import (
    TrainHistory, evaluate, evaluate_adversarial, save_full_checkpoint,
    save_weights_only, train_one_epoch_adversarial,
)
from sar_atr.models import build_model, build_param_groups, count_parameters
from sar_atr.train import build_scheduler, resolve_recipe
from sar_atr.utils import (
    append_csv_row, cuda_memory_summary, cuda_peak_memory_gb, get_logger,
    reset_cuda_peak_stats, save_json, seed_everything, select_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PGD adversarial training of a SAR ATR classifier.",
    )
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    p.add_argument("--dataset", default="atrnet_star", choices=SUPPORTED_DATASETS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=30,
                   help="AT converges in fewer epochs than clean training.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--layer_decay", type=float, default=None)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Defaults to checkpoints/{dataset}/{model}_at_eps{eps}/seed_{seed}/.")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--atrnet_config", default="SOC-40")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--summary_csv", type=Path, default=None)
    # Adversarial-training hyperparameters
    p.add_argument("--at_eps", type=float, default=8 / 255,
                   help="Training L-inf budget in pixel space (default 8/255).")
    p.add_argument("--at_steps", type=int, default=7)
    p.add_argument("--at_alpha", type=float, default=None,
                   help="Inner PGD step size; defaults to 2.5*eps/steps.")
    p.add_argument("--early_stop_patience", type=int, default=10,
                   help="Epochs without robust-val improvement before stopping.")
    p.add_argument("--val_attack_steps", type=int, default=10,
                   help="PGD steps for the per-epoch robust validation.")
    p.add_argument("--robust_val_size", type=int, default=2000,
                   help="Fixed random subset of val used for robust validation "
                        "(caps the per-epoch evaluation cost).")
    return p.parse_args()


def _robust_val_loader(val_loader, size: int, seed: int):
    ds = val_loader.dataset
    if size >= len(ds):
        return val_loader
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=gen)[:size].tolist()
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, sorted(idx)),
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=val_loader.pin_memory,
    )


def main() -> int:
    args = parse_args()

    at_alpha = resolve_pgd_alpha(args.at_eps, args.at_steps, args.at_alpha)
    out_dir: Path = (
        args.output_dir
        if args.output_dir is not None
        else run_dir(args.dataset, f"{args.model}_at_eps{args.at_eps:.4f}", args.seed)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(
        f"advtrain.{args.model}.s{args.seed}.e{args.at_eps:.3f}",
        log_file=out_dir / "adv_train.log",
    )
    logger.info("args = %s", json.dumps(vars(args), default=str))
    logger.info("at_eps=%.5f at_steps=%d at_alpha=%.5f", args.at_eps, args.at_steps, at_alpha)

    recipe = resolve_recipe(args)
    logger.info("recipe = %s", json.dumps(recipe))

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
    logger.info(
        "dataset=%s | train=%d val=%d test=%d classes=%d",
        args.dataset, len(data.train.dataset), len(data.val.dataset),
        len(data.test.dataset), data.num_classes,
    )
    robust_val = _robust_val_loader(data.val, args.robust_val_size, seed=0)
    logger.info("robust val subset=%d", len(robust_val.dataset))

    model = build_model(
        args.model, num_classes=data.num_classes, pretrained=not args.no_pretrained,
    ).to(device)
    logger.info("model=%s | params=%d", args.model, count_parameters(model))

    criterion = nn.CrossEntropyLoss(label_smoothing=recipe["label_smoothing"])
    param_groups = build_param_groups(
        model, args.model,
        base_lr=recipe["lr"],
        weight_decay=recipe["weight_decay"],
        layer_decay=recipe["layer_decay"],
    )
    optimizer = torch.optim.AdamW(param_groups, lr=recipe["lr"])
    scheduler = build_scheduler(optimizer, args.epochs, recipe["warmup_epochs"])

    train_attack = partial(
        pgd_linf, model,
        epsilon=args.at_eps, alpha=at_alpha, steps=args.at_steps,
    )
    val_attack = partial(
        pgd_linf, model,
        epsilon=args.at_eps,
        alpha=resolve_pgd_alpha(args.at_eps, args.val_attack_steps, None),
        steps=args.val_attack_steps,
    )

    history = TrainHistory()
    robust_val_history: list[float] = []
    best_robust_val = 0.0
    patience_counter = 0
    start_epoch = 0

    resume_path = out_dir / "resume_checkpoint.pth"
    if args.resume and resume_path.exists():
        logger.info("resuming from %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_robust_val = ckpt.get("best_val_acc", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        hdict = ckpt.get("history", {})
        history.train_loss = list(hdict.get("train_loss", []))
        history.train_acc = list(hdict.get("train_acc", []))
        history.val_loss = list(hdict.get("val_loss", []))
        history.val_acc = list(hdict.get("val_acc", []))
        robust_val_history = list(ckpt.get("robust_val_history", []))
        logger.info("resumed at epoch=%d best_robust_val=%.4f", start_epoch, best_robust_val)

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_res = train_one_epoch_adversarial(
            model, data.train, criterion, optimizer, device,
            attack_fn=train_attack,
            use_amp=not args.no_amp,
            desc=f"epoch {epoch+1}/{args.epochs} AT train",
        )
        val_res = evaluate(
            model, data.val, criterion, device,
            desc=f"epoch {epoch+1}/{args.epochs} clean val",
        )
        robust_val_acc = evaluate_adversarial(
            model, robust_val, val_attack, device,
            desc=f"epoch {epoch+1}/{args.epochs} robust val",
        )
        scheduler.step()
        history.append(train_res, val_res)
        robust_val_history.append(float(robust_val_acc))

        logger.info(
            "epoch=%02d/%d | adv_train_loss=%.4f adv_train_acc=%.4f | "
            "clean_val_acc=%.4f robust_val_acc=%.4f",
            epoch + 1, args.epochs,
            train_res.loss, train_res.accuracy,
            val_res.accuracy, robust_val_acc,
        )

        improved = robust_val_acc > best_robust_val
        if improved:
            best_robust_val = robust_val_acc
            patience_counter = 0
            save_weights_only(out_dir / "best_model.pth", model)
            logger.info("new best robust val acc %.4f -> saved best_model.pth", best_robust_val)
        else:
            patience_counter += 1

        save_full_checkpoint(
            out_dir / "resume_checkpoint.pth",
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val_acc=best_robust_val,
            patience_counter=patience_counter, history=history,
            class_names=data.class_names,
            extra={
                "dataset": args.dataset, "model_name": args.model,
                "seed": args.seed, "at_eps": args.at_eps,
                "at_steps": args.at_steps, "at_alpha": at_alpha,
                "robust_val_history": robust_val_history,
            },
        )

        if patience_counter >= args.early_stop_patience:
            logger.info("early stopping at epoch %d (no robust-val improvement)", epoch + 1)
            break

    save_weights_only(out_dir / "final_model.pth", model)
    hdict = history.to_dict()
    hdict["robust_val_acc"] = robust_val_history
    save_json(out_dir / "history.json", hdict)

    # Final test evaluation with the best (robust-val-selected) checkpoint:
    # clean accuracy plus a PGD-20 robust check at the trained budget. The
    # full attack grid and AutoAttack run separately via sar-atr-attack.
    best_state = torch.load(out_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_state)
    test_res = evaluate(model, data.test, criterion, device, desc="final clean test")
    test_attack = partial(
        pgd_linf, model,
        epsilon=args.at_eps,
        alpha=resolve_pgd_alpha(args.at_eps, 20, None),
        steps=20,
    )
    robust_test_acc = evaluate_adversarial(
        model, data.test, test_attack, device, desc="final robust test",
    )
    elapsed_sec = time.time() - start_time

    peak_gb = cuda_peak_memory_gb()
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "num_classes": data.num_classes,
        "at_eps": float(args.at_eps),
        "at_steps": int(args.at_steps),
        "at_alpha": float(at_alpha),
        "epochs_trained": len(history.train_loss),
        "best_robust_val_acc": float(best_robust_val),
        "test_acc": float(test_res.accuracy),
        "robust_test_acc_pgd20": float(robust_test_acc),
        "elapsed_sec": float(elapsed_sec),
        "recipe": recipe,
        "peak_vram_gb": peak_gb,
        "class_names": data.class_names,
    }
    save_json(out_dir / "metrics.json", summary)
    logger.info(
        "done | best_robust_val=%.4f clean_test=%.4f robust_test(pgd20@%.4f)=%.4f "
        "elapsed=%.1fs peak_vram=%s",
        best_robust_val, test_res.accuracy, args.at_eps, robust_test_acc, elapsed_sec,
        f"{peak_gb:.1f}GB" if peak_gb is not None else "n/a",
    )

    if args.summary_csv is not None:
        append_csv_row(
            args.summary_csv,
            {k: summary[k] for k in summary if k not in ("class_names", "recipe")},
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
