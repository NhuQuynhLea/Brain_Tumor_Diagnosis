"""Paper results: merge all runs into multi-seed tables + figures (tasks 6.10-6.14).

Reads results/experiments.csv + results/runs/<run_id>/metrics/*.json|csv and produces
in --out dir (default results/paper):
  - per_run_results.csv        every evaluate-on-test row (model, variant, seed, metrics)
  - summary_mean_std.csv       mean +/- std per (model, variant, epochs) across seeds
  - freeze_vs_finetune.csv/png frozen vs fine-tuned F1 per transfer model
  - accuracy_efficiency.png    F1 (mean +/- std) vs GFLOPs, best variant per model
  - per_class_f1.png           per-class F1 grouped bars, best variant per model
  - <model>_confusion_norm.png row-normalized confusion matrix, best run per model

    python scripts/paper_figures.py
    python scripts/paper_figures.py --out results/paper
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.config import load_config
from src.utils.tracking import get_logger

logger = get_logger("paper_figures", Path("results/logs/paper_figures.log"))

METRICS = ["accuracy", "f1_macro", "auc_macro"]


def variant_of(row) -> str:
    if row["model"] == "custom_cnn":
        return "scratch"
    return "freeze" if bool(row["freeze_backbone"]) else "finetune"


def load_eval_rows(exp_path: Path, runs_root: Path) -> pd.DataFrame:
    exp = pd.read_csv(exp_path)
    df = exp[(exp["source"] == "evaluate") & (exp["split"] == "test")].copy()
    df["variant"] = df.apply(variant_of, axis=1)
    df["metrics_json"] = df.apply(
        lambda r: runs_root / str(r["run_id"]) / "metrics" / f"{r['model']}_test_metrics.json",
        axis=1)
    missing = df[~df["metrics_json"].apply(lambda p: p.exists())]
    for _, r in missing.iterrows():
        logger.warning("%s @ %s: metrics dir not synced, skipped", r["model"], r["run_id"])
    df = df[~df.index.isin(missing.index)]
    df[METRICS] = df[METRICS] * 100  # work in percent
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["model", "variant", "epochs"])
             .agg(n=("accuracy", "size"),
                  acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                  f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
                  auc_mean=("auc_macro", "mean"), auc_std=("auc_macro", "std"))
             .reset_index()
             .sort_values("f1_mean", ascending=False))
    return agg.fillna(0.0)


def load_complexity(runs_root: Path) -> dict:
    comp = {}
    for cpath in sorted(runs_root.glob("*/metrics/complexity.csv")):
        for name, row in pd.read_csv(cpath, index_col=0).iterrows():
            comp.setdefault(name, row)
    return comp


def best_group_rows(df: pd.DataFrame, group_row) -> pd.DataFrame:
    return df[(df["model"] == group_row["model"])
              & (df["variant"] == group_row["variant"])
              & (df["epochs"] == group_row["epochs"])]


def plot_accuracy_efficiency(best: pd.DataFrame, comp: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for _, r in best.iterrows():
        c = comp.get(r["model"])
        if c is None:
            continue
        label = f"{r['model']} ({r['variant']})"
        ax.errorbar(c["gflops"], r["f1_mean"], yerr=r["f1_std"], fmt="o", capsize=4)
        ax.annotate(label, (c["gflops"], r["f1_mean"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xlabel("GFLOPs"); ax.set_ylabel("Test F1 macro (%)")
    ax.set_title("Accuracy–Efficiency Trade-off (best variant per model)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "accuracy_efficiency.png", dpi=150); plt.close(fig)


def plot_per_class_f1(df: pd.DataFrame, best: pd.DataFrame, classes: list[str],
                      out: Path) -> None:
    rows = {}
    for _, r in best.iterrows():
        f1s = []
        for _, row in best_group_rows(df, r).iterrows():
            m = json.loads(Path(row["metrics_json"]).read_text())
            f1s.append([m[f"{c}_f1"] * 100 for c in classes])
        rows[f"{r['model']}-{r['variant']}"] = np.mean(f1s, axis=0)
    data = pd.DataFrame(rows, index=classes).round(2)
    data.to_csv(out / "per_class_f1.csv")
    ax = data.T.plot(kind="bar", figsize=(9, 5), width=0.8)
    ax.set_ylabel("F1 (%)"); ax.set_xlabel("Model (best variant)")
    ax.set_title("Per-class F1 (mean over seeds)")
    ax.set_ylim(85, 100); ax.legend(title="class"); plt.xticks(rotation=20, ha="right")
    ax.figure.tight_layout(); ax.figure.savefig(out / "per_class_f1.png", dpi=150)
    plt.close(ax.figure)


def plot_normalized_cms(df: pd.DataFrame, best: pd.DataFrame, classes: list[str],
                        runs_root: Path, out: Path) -> None:
    for _, r in best.iterrows():
        top = best_group_rows(df, r).sort_values("f1_macro", ascending=False).iloc[0]
        pred_path = runs_root / str(top["run_id"]) / "metrics" / f"{top['model']}_predictions.csv"
        if not pred_path.exists():
            logger.warning("no predictions for %s in %s", top["model"], top["run_id"])
            continue
        preds = pd.read_csv(pred_path)
        cm = confusion_matrix(preds["label_idx"], preds["y_pred"],
                              labels=list(range(len(classes))), normalize="true")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    vmin=0, vmax=1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{top['model']} ({top['variant']}, seed {int(top['seed'])}) — normalized")
        fig.tight_layout()
        fig.savefig(out / f"{top['model']}_confusion_norm.png", dpi=150)
        plt.close(fig)


def plot_freeze_vs_finetune(agg: pd.DataFrame, out: Path) -> None:
    ft = agg[agg["variant"].isin(["freeze", "finetune"])]
    f1 = ft.pivot_table(index="model", columns="variant", values="f1_mean")
    std = ft.pivot_table(index="model", columns="variant", values="f1_std")
    merged = ft.pivot_table(index="model", columns="variant",
                            values=["acc_mean", "f1_mean", "auc_mean", "n"])
    merged.to_csv(out / "freeze_vs_finetune.csv")
    x = np.arange(len(f1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, f1["freeze"], 0.4, yerr=std["freeze"], capsize=4, label="freeze")
    ax.bar(x + 0.2, f1["finetune"], 0.4, yerr=std["finetune"], capsize=4, label="fine-tune")
    ax.set_xticks(x); ax.set_xticklabels(f1.index)
    ax.set_ylabel("Test F1 macro (%)"); ax.set_ylim(85, 100)
    ax.set_title("Freeze vs Fine-tune (mean ± std over seeds)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "freeze_vs_finetune.png", dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", default="results/experiments.csv")
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--out", default="results/paper")
    args = p.parse_args()

    cfg = load_config()
    classes = list(cfg.data.classes)
    out, runs_root = Path(args.out), Path(args.runs)
    out.mkdir(parents=True, exist_ok=True)

    df = load_eval_rows(Path(args.experiments), runs_root)
    df.to_csv(out / "per_run_results.csv", index=False)
    logger.info("%d evaluated runs across %d models", len(df), df["model"].nunique())

    agg = summarize(df)
    agg.to_csv(out / "summary_mean_std.csv", index=False, float_format="%.3f")
    logger.info("\n%s", agg.to_string(index=False))

    best = agg.loc[agg.groupby("model")["f1_mean"].idxmax()]
    comp = load_complexity(runs_root)

    plot_accuracy_efficiency(best, comp, out)
    plot_per_class_f1(df, best, classes, out)
    plot_normalized_cms(df, best, classes, runs_root, out)
    plot_freeze_vs_finetune(agg, out)
    logger.info("Wrote paper artifacts to %s", out)


if __name__ == "__main__":
    main()
