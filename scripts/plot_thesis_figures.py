from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ATTACK_DIR = RESULTS_DIR / "attacks" / "atrnet_star"
TRAIN_CSV = RESULTS_DIR / "train_summary.csv"
OUT_DIR = ROOT / "img"

MODEL_ORDER = ["resnet50", "efficientnet_b3", "vit_b_16"]
MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b3": "EfficientNet-B3",
    "vit_b_16": "ViT-B/16",
}
ATTACK_LABELS = {"fgsm": "FGSM", "pgd": "PGD", "cw": "CW"}
CHANCE_ACC = 100 / 40


def set_apa_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 12,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600)
    plt.close(fig)


def load_attack_results() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in sorted(ATTACK_DIR.rglob("*.csv")):
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows.extend(reader)

    if not rows:
        raise FileNotFoundError(f"No attack CSV files found under {ATTACK_DIR}")

    df = pd.DataFrame(rows)
    numeric_cols = [
        "seed",
        "epsilon",
        "clean_acc",
        "adv_acc",
        "attack_success_rate",
        "attack_sec",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df["accuracy_pct"] = df["adv_acc"] * 100
    df["clean_accuracy_pct"] = df["clean_acc"] * 100
    return df.sort_values(["model", "seed", "attack_type", "epsilon"])


def load_training_results() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df["test_accuracy_pct"] = df["test_acc"] * 100
    return df.sort_values(["model", "seed"])


def summarize_mean_sd(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_cols, observed=True)[value_col]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )


def plot_training_accuracy(train_df: pd.DataFrame) -> None:
    summary = summarize_mean_sd(train_df, ["model"], "test_accuracy_pct")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(summary))
    means = summary["mean"].to_numpy()
    sds = summary["sd"].to_numpy()

    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=4,
        color=["0.20", "0.45", "0.70"],
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.axhline(CHANCE_ACC, color="black", linestyle=":", linewidth=1.0)
    ax.text(2.45, CHANCE_ACC + 1.4, "Chance = 2.5%", ha="right", va="bottom", fontsize=9)

    for bar, mean in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 1.5,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x, [MODEL_LABELS[str(model)] for model in summary["model"]])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Architecture")
    ax.set_ylim(0, 105)
    ax.set_title("Clean ATRNet-STAR Classification Accuracy")
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, "atrnet_star_clean_accuracy_apa")


def plot_attack_curves(attack_df: pd.DataFrame) -> None:
    curve_df = attack_df[attack_df["attack_type"].isin(["fgsm", "pgd"])].copy()
    summary = summarize_mean_sd(
        curve_df,
        ["model", "attack_type", "epsilon"],
        "accuracy_pct",
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    styles = {
        "fgsm": {"color": "black", "marker": "o", "linestyle": "-", "label": "FGSM"},
        "pgd": {"color": "0.45", "marker": "s", "linestyle": "--", "label": "PGD"},
    }

    for idx, (ax, model) in enumerate(zip(axes, MODEL_ORDER, strict=True)):
        model_summary = summary[summary["model"] == model]
        for attack_type in ["fgsm", "pgd"]:
            rows = model_summary[model_summary["attack_type"] == attack_type].sort_values("epsilon")
            ax.errorbar(
                rows["epsilon"],
                rows["mean"],
                yerr=rows["sd"],
                capsize=3,
                linewidth=1.4,
                markersize=4.5,
                elinewidth=0.8,
                capthick=0.8,
                **styles[attack_type],
            )

        clean_acc = (
            attack_df[attack_df["model"] == model]
            .groupby("seed", observed=True)["clean_accuracy_pct"]
            .first()
            .mean()
        )
        ax.axhline(clean_acc, color="0.25", linestyle="-.", linewidth=0.9)
        ax.axhline(CHANCE_ACC, color="black", linestyle=":", linewidth=0.9)

        if idx == 0:
            ax.set_ylabel("Adversarial accuracy (%)")
        ax.set_xlabel("Perturbation budget (epsilon)")
        ax.set_title(MODEL_LABELS[model])
        ax.set_xticks([0.01, 0.02, 0.05, 0.10], ["0.01", "0.02", "0.05", "0.10"])
        ax.set_ylim(0, 105)
        clean_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("FGSM and PGD Robustness Curves on ATRNet-STAR", y=1.12)
    fig.tight_layout()
    save_figure(fig, "atrnet_star_fgsm_pgd_curves_apa")


def plot_cw_results(attack_df: pd.DataFrame) -> None:
    clean_df = (
        attack_df.groupby(["model", "seed"], observed=True)["clean_accuracy_pct"]
        .first()
        .reset_index()
        .assign(condition="Clean", accuracy_pct=lambda df: df["clean_accuracy_pct"])
    )
    cw_df = (
        attack_df[attack_df["attack_type"] == "cw"]
        .loc[:, ["model", "seed", "accuracy_pct"]]
        .assign(condition="CW")
    )
    combined = pd.concat(
        [
            clean_df.loc[:, ["model", "seed", "condition", "accuracy_pct"]],
            cw_df.loc[:, ["model", "seed", "condition", "accuracy_pct"]],
        ],
        ignore_index=True,
    )
    combined["condition"] = pd.Categorical(combined["condition"], ["Clean", "CW"], ordered=True)
    summary = summarize_mean_sd(combined, ["model", "condition"], "accuracy_pct")

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    offsets = {"Clean": -width / 2, "CW": width / 2}
    colors = {"Clean": "0.25", "CW": "0.70"}

    for condition in ["Clean", "CW"]:
        rows = summary[summary["condition"] == condition].set_index("model").loc[MODEL_ORDER]
        ax.bar(
            x + offsets[condition],
            rows["mean"],
            width=width,
            yerr=rows["sd"],
            capsize=4,
            label=condition,
            color=colors[condition],
            edgecolor="black",
            linewidth=0.8,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )

    ax.axhline(CHANCE_ACC, color="black", linestyle=":", linewidth=1.0)
    ax.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Architecture")
    ax.set_ylim(0, 105)
    ax.set_title("Clean Accuracy Compared With CW Attack Accuracy")
    ax.legend(frameon=False, loc="lower left")
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, "atrnet_star_cw_accuracy_apa")


def write_summary_tables(train_df: pd.DataFrame, attack_df: pd.DataFrame) -> None:
    train_summary = summarize_mean_sd(train_df, ["model"], "test_accuracy_pct")
    attack_summary = summarize_mean_sd(
        attack_df,
        ["model", "attack_type", "epsilon"],
        "accuracy_pct",
    )

    train_summary["model"] = train_summary["model"].astype(str).map(MODEL_LABELS)
    attack_summary["model"] = attack_summary["model"].astype(str).map(MODEL_LABELS)
    attack_summary["attack_type"] = attack_summary["attack_type"].map(ATTACK_LABELS)

    train_summary.to_csv(OUT_DIR / "atrnet_star_clean_accuracy_summary.csv", index=False)
    attack_summary.to_csv(OUT_DIR / "atrnet_star_attack_accuracy_summary.csv", index=False)


def main() -> None:
    set_apa_style()
    train_df = load_training_results()
    attack_df = load_attack_results()

    plot_training_accuracy(train_df)
    plot_attack_curves(attack_df)
    plot_cw_results(attack_df)
    write_summary_tables(train_df, attack_df)

    outputs = sorted(OUT_DIR.glob("atrnet_star_*_apa.*")) + sorted(
        OUT_DIR.glob("atrnet_star_*_summary.csv")
    )
    print("Generated thesis figures and summary tables:")
    for path in outputs:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
