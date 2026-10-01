"""Aggregate results across trained models (Phase 6).

Reads models/<name>/history.csv + results/metrics/<name>_predictions.csv and produces:
  - results/metrics/comparison_table.csv   (all metrics, all models)
  - results/metrics/complexity.csv         (params, GFLOPs, size, inference ms)
  - results/metrics/mcnemar.txt            (statistical test between top-2 models)
  - results/metrics/final_recommendation.txt
  - results/figures/training_curves_all.png, roc_all.png, error_analysis.png

    python scripts/compare_models.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import chi2

from src.evaluation.complexity import profile_model
from src.evaluation.plots import plot_roc_curves
from src.models.model_factory import available_models, build_model
from src.utils.config import load_config
from src.utils.runs import resolve_run, use_run
from src.utils.tracking import add_log_file, get_logger

logger = get_logger("compare", Path("results/logs/compare.log"))


def find_trained_models(cfg) -> list[str]:
    return [p.parent.name for p in Path(cfg.paths.checkpoints).glob("*/best.pt")]


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar's test with continuity correction on per-sample correctness."""
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p_value = float(1 - chi2.cdf(stat, df=1)) if (b + c) > 0 else 1.0
    return {"b": b, "c": c, "chi2": round(stat, 4), "p_value": round(p_value, 6),
            "significant": p_value < 0.05}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="run_id, run name, or path (default: latest run)")
    args = p.parse_args()

    cfg = load_config()
    run_dir = use_run(cfg, resolve_run(cfg, args.run))
    add_log_file(logger, run_dir / "logs" / "compare.log")
    logger.info("Run dir: %s", run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = find_trained_models(cfg)
    if not models:
        logger.error("No trained checkpoints found in %s", cfg.paths.checkpoints)
        sys.exit(1)
    logger.info("Found trained models: %s", models)

    metrics_dir, fig_dir = Path(cfg.paths.metrics), Path(cfg.paths.figures)
    classes = list(cfg.data.classes)

    # --- per-model metrics from predictions CSVs (written by evaluate.py) ---
    all_metrics, preds = {}, {}
    for name in models:
        pred_csv = metrics_dir / f"{name}_predictions.csv"
        if not pred_csv.exists():
            logger.warning("%s: no predictions file, run scripts/evaluate.py --model %s", name, name)
            continue
        df = pd.read_csv(pred_csv)
        y_true = df["label_idx"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        y_prob = df[[f"prob_{c}" for c in classes]].to_numpy()
        from src.evaluation.metrics import compute_metrics
        all_metrics[name] = compute_metrics(y_true, y_pred, classes)
        aucs = plot_roc_curves(y_true, y_prob, classes, fig_dir / f"{name}_roc.png",
                               title=f"{name} - ROC")
        all_metrics[name]["auc_macro"] = aucs["macro"]
        preds[name] = df.set_index("filepath")

    if not all_metrics:
        sys.exit("Nothing to compare - run evaluate.py first.")

    # --- 6.3 comparison table ---
    from src.evaluation.metrics import flatten_metrics_table
    table = flatten_metrics_table(all_metrics, classes)
    table.insert(0, "auc_macro", [all_metrics[m].get("auc_macro") for m in table.index])
    table.to_csv(metrics_dir / "comparison_table.csv")
    logger.info("Comparison table:\n%s", table.to_string())

    # --- 6.2 combined training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in models:
        hist_path = Path(cfg.paths.checkpoints) / name / "history.csv"
        if hist_path.exists():
            h = pd.read_csv(hist_path)
            axes[0].plot(h["epoch"], h["val_loss"], label=name)
            axes[1].plot(h["epoch"], h["val_f1_macro"], label=name)
    axes[0].set_title("Validation Loss"); axes[1].set_title("Validation F1 (macro)")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "training_curves_all.png", dpi=150); plt.close(fig)

    # --- 6.7 error analysis: which classes are hardest overall ---
    err_rows = []
    for name, df in preds.items():
        err = df[~df["correct"]]
        for cls in classes:
            sub = err[err["label"] == cls]
            err_rows.append({"model": name, "true_class": cls, "errors": len(sub)})
    err_df = pd.DataFrame(err_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=err_df, x="true_class", y="errors", hue="model", ax=ax)
    ax.set_title("Misclassified images per class"); fig.tight_layout()
    fig.savefig(fig_dir / "error_analysis.png", dpi=150); plt.close(fig)
    err_df.to_csv(metrics_dir / "error_analysis.csv", index=False)

    # --- 6.8 complexity ---
    complexity = {}
    for name in models:
        model = build_model(cfg, name=name)
        ckpt = Path(cfg.paths.checkpoints) / name / "best.pt"
        complexity[name] = profile_model(model, device=torch.device(device),
                                         image_size=cfg.data.image_size,
                                         checkpoint_path=ckpt)
    comp_df = pd.DataFrame(complexity).T
    comp_df.to_csv(metrics_dir / "complexity.csv")
    logger.info("Complexity:\n%s", comp_df.to_string())

    # --- 6.5 McNemar between top-2 by f1_macro ---
    ranked = sorted(all_metrics, key=lambda m: all_metrics[m]["f1_macro"], reverse=True)
    lines = []
    if len(ranked) >= 2 and all(n in preds for n in ranked[:2]):
        a, b = ranked[0], ranked[1]
        common = preds[a].join(preds[b][["correct"]], rsuffix="_b", how="inner")
        res = mcnemar_test(common["correct"].to_numpy(), common["correct_b"].to_numpy())
        lines.append(f"McNemar {a} vs {b}: {res}")
        with open(metrics_dir / "mcnemar.txt", "w") as f:
            f.write("\n".join(lines))
        logger.info(lines[0])

    # --- 6.9 recommendation ---
    best = ranked[0]
    rec = (
        f"Best model by val/test F1(macro): {best}\n"
        f"  accuracy={all_metrics[best]['accuracy']:.4f}  f1={all_metrics[best]['f1_macro']:.4f}  "
        f"auc={all_metrics[best].get('auc_macro', float('nan')):.4f}\n"
        f"  params={comp_df.loc[best, 'params_total']:.0f}  "
        f"gflops={comp_df.loc[best, 'gflops']}  inference={comp_df.loc[best, 'inference_ms']}ms/img\n"
        f"Ranking by f1_macro: {', '.join(ranked)}"
    )
    with open(metrics_dir / "final_recommendation.txt", "w") as f:
        f.write(rec)
    logger.info("\n%s", rec)


if __name__ == "__main__":
    main()
