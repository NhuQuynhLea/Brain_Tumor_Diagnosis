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
INK = "#203047"
MUTED = "#526277"
GRID = "#DFE5EB"
COLORS = {"freeze": "#0072B2", "finetune": "#D55E00", "scratch": "#009E73"}
MARKERS = {"freeze": "o", "finetune": "D", "scratch": "s"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def save_chart(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def variant_of(row) -> str:
    if row["model"] in {"custom_cnn", "custom_cnn_tiny", "lsnet", "resnet_tiny",
                        "mpac_resnet_tiny", "efficientnet_tiny", "mobilenet_tiny"}:
        return "scratch"
    return "freeze" if bool(row["freeze_backbone"]) else "finetune"


def load_eval_rows(exp_path: Path, runs_root: Path, classes: list[str]) -> pd.DataFrame:
    exp = pd.read_csv(exp_path)
    df = exp[(exp["source"] == "evaluate") & (exp["split"] == "test")].copy()
    df = df.drop_duplicates(["run_id", "model", "source", "split"], keep="last")
    df["variant"] = df.apply(variant_of, axis=1)
    df["metrics_json"] = df.apply(
        lambda r: runs_root / str(r["run_id"]) / "metrics" / f"{r['model']}_test_metrics.json",
        axis=1)
    missing = df[~df["metrics_json"].apply(lambda p: p.exists())]
    for _, r in missing.iterrows():
        logger.warning("%s @ %s: metrics dir not synced, skipped", r["model"], r["run_id"])
    df = df[~df.index.isin(missing.index)]
    valid = df["metrics_json"].apply(
        lambda path: all(f"{c}_f1" in json.loads(path.read_text()) for c in classes))
    logger.info("Excluded %d runs without all %s class metrics", int((~valid).sum()), classes)
    df = df[valid].copy()
    compact = {"lsnet", "mpac_resnet_tiny", "efficientnet_tiny",
               "mobilenet_tiny", "custom_cnn_tiny"}
    df = df[~df["model"].isin(compact) | df["early_stopping_patience"].eq(15)].copy()
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
    paths = sorted(runs_root.glob("*/metrics/complexity.csv"))
    global_comp = Path("results/metrics/complexity_all.csv")  # from profile_models.py
    if global_comp.exists():
        paths.append(global_comp)
    for cpath in paths:
        for name, row in pd.read_csv(cpath, index_col=0).iterrows():
            comp.setdefault(name, row)
    return comp


def best_group_rows(df: pd.DataFrame, group_row) -> pd.DataFrame:
    return df[(df["model"] == group_row["model"])
              & (df["variant"] == group_row["variant"])
              & (df["epochs"] == group_row["epochs"])]


def plot_accuracy_efficiency(best: pd.DataFrame, comp: dict, out: Path) -> None:
    points = [(r, float(comp[r["model"]]["gflops"])) for _, r in best.iterrows()
              if r["model"] in comp and np.isfinite(comp[r["model"]]["gflops"])]
    points.sort(key=lambda item: (item[1], item[0]["model"]))
    if not points:
        logger.warning("no complexity data available for accuracy-efficiency figure")
        return
    fig, (ax, key) = plt.subplots(1, 2, figsize=(11.5, 6.5),
                                   gridspec_kw={"width_ratios": [3.1, 1.5]})
    ys = np.array([float(r["f1_mean"]) for r, _ in points])
    errors = np.array([float(r["f1_std"]) for r, _ in points])
    xs = np.array([x for _, x in points])
    span = max(float(np.ptp(ys + errors)), float(np.ptp(ys - errors)), 1)
    ax.set_ylim(max(0, float(np.min(ys - errors)) - .12 * span),
                max(100, float(np.max(ys + errors)) + .18 * span))
    ax.set_xscale("log")
    ax.set_xlim(float(xs.min() / 1.4), float(xs.max() * 1.4))
    ax.grid(axis="both", color=GRID, linewidth=.8, zorder=0)
    for i, (r, x) in enumerate(points, 1):
        color = COLORS[r["variant"]]
        ax.errorbar(x, r["f1_mean"], yerr=r["f1_std"], fmt="none",
                    ecolor=color, elinewidth=1.5, capsize=3, zorder=2)
        ax.scatter(x, r["f1_mean"], marker=MARKERS[r["variant"]], s=250,
                   color=color, edgecolor="white", linewidth=1.3, zorder=3)
        ax.text(x, r["f1_mean"], chr(64 + i), ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=4)
    ax.set_xlabel("Compute per image (GFLOPs, log scale)")
    ax.set_ylabel("Test macro F1 (%)")
    ax.set_title("Performance vs. compute  |  zoomed F1 axis", loc="left", pad=15)
    key.axis("off")
    key.set_title("MODEL KEY  ·  BEST VARIANT", loc="left", fontsize=9, color=MUTED, pad=15)
    for i, (r, _) in enumerate(points, 1):
        y = 1 - (i - .5) / len(points)
        key.text(.01, y, chr(64 + i), transform=key.transAxes, va="center",
                 fontsize=11, fontweight="bold", color=COLORS[r["variant"]])
        key.text(.15, y, f"{r['model'].replace('_', '-')}  ·  {r['variant']}", transform=key.transAxes,
                 va="center", fontsize=10)
    fig.text(.07, .02, "Nickparvar four-class · ±1 SD across logged runs; one-run groups have no SD.",
             color=MUTED, fontsize=10.5)
    fig.tight_layout(rect=(0, .055, 1, 1), w_pad=2)
    save_chart(fig, out, "accuracy_efficiency")


def plot_per_class_f1(df: pd.DataFrame, best: pd.DataFrame, classes: list[str],
                      out: Path, write_csv: bool = True) -> None:
    rows = {}
    sample_counts = {}
    for _, r in best.iterrows():
        f1s = []
        for _, row in best_group_rows(df, r).iterrows():
            m = json.loads(Path(row["metrics_json"]).read_text())
            if all(f"{c}_f1" in m for c in classes):
                f1s.append([m[f"{c}_f1"] * 100 for c in classes])
        if f1s:
            label = f"{r['model']}-{r['variant']}"
            rows[label] = np.mean(f1s, axis=0)
            sample_counts[label] = len(f1s)
        else:
            logger.warning("no class-level metrics for %s / %s", r["model"], r["variant"])
    data = pd.DataFrame(rows, index=classes).round(2)
    if write_csv:
        data.to_csv(out / "per_class_f1.csv")
    order = best.sort_values("f1_mean", ascending=False)
    matrix = data[[f"{r['model']}-{r['variant']}" for _, r in order.iterrows()
                   if f"{r['model']}-{r['variant']}" in data.columns]].T
    fig, ax = plt.subplots(figsize=(9.8, max(5, .49 * len(matrix) + 1.8)))
    cmap = sns.light_palette(COLORS["freeze"], as_cmap=True)
    lower = max(0, float(np.floor(matrix.min().min())))
    upper = min(100, float(np.ceil(matrix.max().max())))
    if lower == upper:
        lower = max(0, lower - 1)
    sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=lower, vmax=upper, linewidths=1,
                linecolor="white", cbar_kws={"label": "Macro F1 by class (%)", "shrink": .75})
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            rgba = cmap(np.clip((value - lower) / (upper - lower), 0, 1))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(j + .5, i + .5, f"{value:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="medium", color="white" if luminance < .52 else INK)
    ax.set_xticklabels([c.replace("notumor", "no tumor").capitalize() for c in classes],
                       rotation=0)
    ax.set_yticklabels([f"{label.replace('-', '  ·  ')}  (n={sample_counts[label]})"
                        for label in matrix.index], rotation=0)
    ax.tick_params(length=0)
    ax.set_xlabel("Diagnosis class", labelpad=12)
    ax.set_ylabel("")
    ax.set_title("Class-level performance", loc="left", pad=17)
    fig.text(.34, .02, "Mean F1 of runs reporting these classes · best variant per model · %",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_chart(fig, out, "per_class_f1")


def plot_normalized_cms(df: pd.DataFrame, best: pd.DataFrame, classes: list[str],
                        runs_root: Path, out: Path) -> None:
    for _, r in best.iterrows():
        top = None
        preds = None
        for _, candidate in best_group_rows(df, r).sort_values("f1_macro", ascending=False).iterrows():
            pred_path = (runs_root / str(candidate["run_id"]) / "metrics"
                         / f"{candidate['model']}_predictions.csv")
            if not pred_path.exists():
                continue
            candidate_preds = pd.read_csv(pred_path)
            if set(candidate_preds["label"].unique()) == set(classes):
                top, preds = candidate, candidate_preds
                break
        if top is None:
            logger.warning("no predictions with expected classes for %s", r["model"])
            continue
        cm = confusion_matrix(preds["label_idx"], preds["y_pred"],
                              labels=list(range(len(classes))), normalize="true")
        counts = np.bincount(preds["label_idx"].to_numpy(dtype=int), minlength=len(classes))
        fig, ax = plt.subplots(figsize=(6.8, 5.8))
        cmap = sns.light_palette(COLORS["freeze"], as_cmap=True)
        sns.heatmap(cm, cmap=cmap, xticklabels=classes, yticklabels=classes, ax=ax,
                    vmin=0, vmax=1, square=True, linewidths=1.5, linecolor="white",
                    cbar_kws={"label": "Share of true class", "shrink": .77,
                               "ticks": [0, .25, .5, .75, 1]})
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] > 0:
                    ax.text(j + .5, i + .5, f"{cm[i, j] * 100:.1f}%",
                            ha="center", va="center", fontsize=10,
                            fontweight="semibold" if i == j else "normal",
                            color="white" if cm[i, j] > .65 else INK)
        ax.set_xticklabels([c.replace("notumor", "no tumor").capitalize() for c in classes],
                           rotation=0)
        ax.set_yticklabels([f"{c.replace('notumor', 'no tumor').capitalize()}  (n={n})"
                            for c, n in zip(classes, counts)], rotation=0)
        ax.tick_params(length=0)
        ax.set_xlabel("Predicted diagnosis", labelpad=12)
        ax.set_ylabel("True diagnosis", labelpad=12)
        ax.set_title(f"{top['model']}  ·  {top['variant']}  ·  seed {int(top['seed'])}",
                     loc="left", pad=15)
        fig.text(.14, .02, "Row-normalized · blank cells = 0% · best matching-class test-F1 run",
                 fontsize=8.5, color=MUTED)
        fig.tight_layout(rect=(0, .04, 1, 1))
        save_chart(fig, out, f"{top['model']}_confusion_norm")


def plot_freeze_vs_finetune(agg: pd.DataFrame, out: Path, write_csv: bool = True) -> None:
    ft = agg[agg["variant"].isin(["freeze", "finetune"])]
    f1 = ft.pivot_table(index="model", columns="variant", values="f1_mean")
    std = ft.pivot_table(index="model", columns="variant", values="f1_std")
    merged = ft.pivot_table(index="model", columns="variant",
                            values=["acc_mean", "f1_mean", "auc_mean", "n"])
    if write_csv:
        merged.to_csv(out / "freeze_vs_finetune.csv")
    models = f1.max(axis=1).sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9.5, max(5.3, .46 * len(models) + 1.8)))
    for i, model in enumerate(models):
        present = f1.loc[model].dropna()
        if len(present) == 2:
            ax.plot(present.values, [i, i], color="#D8DFE5", lw=1.1, zorder=1)
        for variant, offset in (("freeze", -.12), ("finetune", .12)):
            if variant not in present:
                continue
            mean = float(f1.loc[model, variant])
            error = float(std.loc[model, variant])
            ax.errorbar(mean, i + offset, xerr=error, fmt=MARKERS[variant],
                        color=COLORS[variant], markersize=7.5, markeredgecolor="white",
                        markeredgewidth=.8, elinewidth=1.5, capsize=2.5, zorder=3)
    ax.set_yticks(np.arange(len(models)), models)
    ax.invert_yaxis()
    ax.set_xlim(max(0, np.floor(np.nanmin(f1.values - std.values) - 1)),
                max(100, np.ceil(np.nanmax(f1.values + std.values) + 1)))
    ax.set_xlabel("Test macro F1 (%)", labelpad=10)
    ax.set_title("Frozen vs. fine-tuned  |  zoomed F1 axis", loc="left", pad=15)
    ax.grid(axis="x", color=GRID, linewidth=.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=MARKERS[v], linestyle="none", color=COLORS[v],
                      markersize=8, label=label)
               for v, label in (("freeze", "Frozen"), ("finetune", "Fine-tuned"))]
    fig.legend(handles=handles, frameon=False, loc="upper right", bbox_to_anchor=(.97, .98), ncol=2)
    fig.text(.18, .02, "Nickparvar four-class · ±1 SD · no whisker: one run or zero SD · missing: not evaluated",
             fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(0, .055, 1, 1))
    save_chart(fig, out, "freeze_vs_finetune")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", default="results/experiments.csv")
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--out", default="results/paper")
    p.add_argument("--figures-only", action="store_true", help="regenerate charts without rewriting CSV tables")
    args = p.parse_args()

    cfg = load_config()
    classes = list(cfg.data.classes)
    out, runs_root = Path(args.out), Path(args.runs)
    out.mkdir(parents=True, exist_ok=True)

    df = load_eval_rows(Path(args.experiments), runs_root, classes)
    if not args.figures_only:
        df.to_csv(out / "per_run_results.csv", index=False)
    logger.info("%d evaluated runs across %d models", len(df), df["model"].nunique())

    agg = summarize(df)
    if not args.figures_only:
        agg.to_csv(out / "summary_mean_std.csv", index=False, float_format="%.3f")
    logger.info("\n%s", agg.to_string(index=False))

    best = agg.loc[agg.groupby("model")["f1_mean"].idxmax()]
    comp = load_complexity(runs_root)

    plot_accuracy_efficiency(best, comp, out)
    plot_per_class_f1(df, best, classes, out, write_csv=not args.figures_only)
    plot_normalized_cms(df, best, classes, runs_root, out)
    plot_freeze_vs_finetune(agg, out, write_csv=not args.figures_only)
    logger.info("Wrote paper artifacts to %s", out)


if __name__ == "__main__":
    main()
