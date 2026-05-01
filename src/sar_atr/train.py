"""Train a SAR ATR classifier with reproducible, cluster-friendly I/O.

Typical invocation from a SLURM array task (after `uv sync` / `pip install -e .`):

    sar-atr-train \
        --dataset atrnet_star \
        --model resnet50 \
        --seed 0 \
        --epochs 50 \
        --batch_size 64 \
        --data_dir /scratch/$USER/datasets/atrnet_star

Equivalent without console scripts:

    python -m sar_atr.train --dataset atrnet_star ...

Outputs (per run):

    checkpoints/{dataset}/{model}/seed_{seed}/
        best_model.pth           # model weights only; lowest val loss wins ties
        final_model.pth          # last-epoch weights
        resume_checkpoint.pth    # optimizer / scheduler / history (enables --resume)
        metrics.json             # final accuracy, best-val-acc, elapsed time
        history.json             # per-epoch train/val loss + accuracy
        train.log                # copy of stdout
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from sar_atr.config import (
    SUPPORTED_DATASETS, SUPPORTED_MODELS,
    default_model_cache_dir, run_dir,
)
from sar_atr.datasets import load_dataset
from sar_atr.engine import (
    TrainHistory, evaluate, save_full_checkpoint, save_weights_only,
    train_one_epoch,
)
from sar_atr.models import build_model, count_parameters
from sar_atr.utils import (
    append_csv_row, cuda_memory_summary, get_logger, save_json,
    seed_everything, select_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a SAR ATR classifier on MSTAR or ATRNet-STAR.",
    )
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    p.add_argument("--dataset", default="atrnet_star", choices=SUPPORTED_DATASETS)
    p.add_argument("--seed", type=int, required=True,
                   help="Seed for the full pipeline (reproducibility + data split).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--data_dir", type=Path, required=True,
                   help="MSTAR `Padded_imgs/` or ATRNet-STAR archive/config root.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Override the default checkpoints/{dataset}/{model}/seed_{seed}/.")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Dataloader workers. Auto-detected from GPU VRAM if unset.")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--atrnet_config", default="SOC-40",
                   help="ATRNet-STAR experimental configuration folder.")
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help=">1 to fit ViT-B into limited VRAM via gradient accumulation.")
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--no_amp", action="store_true",
                   help="Disable torch.amp mixed precision training.")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Train from random init instead of ImageNet-1K weights.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from resume_checkpoint.pth if present in output_dir.")
    p.add_argument("--summary_csv", type=Path, default=None,
                   help="Append a single summary row to this CSV after training.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_dir: Path = (
        args.output_dir
        if args.output_dir is not None
        else run_dir(args.dataset, args.model, args.seed)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(f"train.{args.model}.s{args.seed}", log_file=out_dir / "train.log")
    logger.info("args = %s", json.dumps(vars(args), default=str))

    seed_everything(args.seed)
    device = select_device()
    logger.info("device=%s | %s", device, cuda_memory_summary())

    # Force torchvision pretrained-weight downloads into a shared project cache
    # so compute nodes don't re-download and can work on networks without
    # internet access.
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

    model = build_model(
        args.model, num_classes=data.num_classes, pretrained=not args.no_pretrained,
    ).to(device)
    logger.info("model=%s | params=%d", args.model, count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    history = TrainHistory()
    best_val_acc = 0.0
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
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        hdict = ckpt.get("history", {})
        history.train_loss = list(hdict.get("train_loss", []))
        history.train_acc = list(hdict.get("train_acc", []))
        history.val_loss = list(hdict.get("val_loss", []))
        history.val_acc = list(hdict.get("val_acc", []))
        logger.info("resumed at epoch=%d best_val_acc=%.4f", start_epoch, best_val_acc)

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_res = train_one_epoch(
            model, data.train, criterion, optimizer, device,
            use_amp=not args.no_amp,
            grad_accum_steps=args.grad_accum_steps,
            desc=f"epoch {epoch+1}/{args.epochs} train",
        )
        val_res = evaluate(
            model, data.val, criterion, device,
            desc=f"epoch {epoch+1}/{args.epochs} val",
        )
        scheduler.step()
        history.append(train_res, val_res)

        logger.info(
            "epoch=%02d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch + 1, args.epochs,
            train_res.loss, train_res.accuracy,
            val_res.loss, val_res.accuracy,
        )

        improved = val_res.accuracy > best_val_acc
        if improved:
            best_val_acc = val_res.accuracy
            patience_counter = 0
            save_weights_only(out_dir / "best_model.pth", model)
            logger.info("new best val acc %.4f -> saved best_model.pth", best_val_acc)
        else:
            patience_counter += 1

        save_full_checkpoint(
            out_dir / "resume_checkpoint.pth",
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val_acc=best_val_acc,
            patience_counter=patience_counter, history=history,
            class_names=data.class_names,
            extra={"dataset": args.dataset, "model_name": args.model, "seed": args.seed},
        )

        if patience_counter >= args.early_stop_patience:
            logger.info("early stopping at epoch %d (no val improvement)", epoch + 1)
            break

    save_weights_only(out_dir / "final_model.pth", model)
    save_json(out_dir / "history.json", history.to_dict())

    # Final test evaluation with best checkpoint.
    best_state = torch.load(out_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_state)
    test_res = evaluate(model, data.test, criterion, device, desc="final test")
    elapsed_sec = time.time() - start_time

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "num_classes": data.num_classes,
        "epochs_trained": len(history.train_loss),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_res.loss),
        "test_acc": float(test_res.accuracy),
        "elapsed_sec": float(elapsed_sec),
        "class_names": data.class_names,
    }
    save_json(out_dir / "metrics.json", summary)
    logger.info(
        "done | best_val_acc=%.4f test_acc=%.4f elapsed=%.1fs",
        best_val_acc, test_res.accuracy, elapsed_sec,
    )

    if args.summary_csv is not None:
        append_csv_row(
            args.summary_csv,
            {k: summary[k] for k in summary if k != "class_names"},
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
