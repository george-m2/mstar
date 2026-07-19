"""Aggregate per-seed attack results into a summary with uncertainty.

With n=5 seeds, a bare +/- SD both under-describes the spread (per-seed
robustness at small epsilon ranges over tens of points) and invites the
small-sample objection. This reports per-cell mean, SD, min, max, the
individual per-seed values, and a bootstrap percentile CI of the mean.

    sar-atr-aggregate \
        --results_dir results/atrnet_star \
        --output results/atrnet_star/atrnet_attksum.csv

Reads every CSV under --results_dir that has the sar-atr-attack row schema
(model, seed, attack_type, epsilon, adv_acc); duplicate (model, seed, attack,
epsilon) rows keep the most recent occurrence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"model", "seed", "attack_type", "epsilon", "adv_acc"}


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def load_results(results_dir: Path) -> pd.DataFrame:
    frames = []
    for csv in sorted(results_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if REQUIRED_COLUMNS.issubset(df.columns):
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No attack-result CSVs (columns {sorted(REQUIRED_COLUMNS)}) under {results_dir}."
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(
        subset=["model", "seed", "attack_type", "epsilon"], keep="last",
    )


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, attack, eps), group in df.groupby(
        ["model", "attack_type", "epsilon"], sort=True,
    ):
        group = group.sort_values("seed")
        acc = group["adv_acc"].to_numpy(dtype=float) * 100.0
        lo, hi = bootstrap_ci(acc) if acc.size > 1 else (float(acc[0]), float(acc[0]))
        rows.append({
            "model": model,
            "attack_type": attack,
            "epsilon": eps,
            "n": int(acc.size),
            "mean": float(acc.mean()),
            "sd": float(acc.std(ddof=1)) if acc.size > 1 else 0.0,
            "min": float(acc.min()),
            "max": float(acc.max()),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "seeds": ";".join(str(int(s)) for s in group["seed"]),
            "per_seed_acc": ";".join(f"{a:.2f}" for a in acc),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarise per-seed attack results with bootstrap CIs.",
    )
    p.add_argument("--results_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarise(load_results(args.results_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"{len(summary)} cells -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
