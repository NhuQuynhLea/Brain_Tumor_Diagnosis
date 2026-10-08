"""Evaluate trained model(s) on a dataset split.

Produces per model:
  - results/metrics/<model>_<split>_metrics.json + appends to model_comparison.csv
  - results/metrics/<model>_predictions.csv (per-sample y_true/y_pred/probs)
  - results/metrics/<model>_errors.csv (misclassified samples)
  - results/figures/<model>_confusion_matrix.png, <model>_roc.png, <model>_gradcam.png

    python scripts/evaluate.py --model resnet50
    python scripts/evaluate.py --model resnet50,efficientnetb0
    python scripts/evaluate.py --model all            # every model with a best.pt
    python scripts/evaluate.py --model resnet50 --split val --max-samples 200
    # checkpoint sweep: evaluate EVERY saved checkpoint (epochs/*.pt + best + last)
    python scripts/evaluate.py --model resnet18 --split test --checkpoints all
    python scripts/evaluate.py --model all --split test --checkpoints all
    # external dataset (task 7.4): metrics/figures get a <dataset>_<csv> tag
    python scripts/evaluate.py --model resnet50 --run <ft_run> --csv data/external/figshare/all.csv --restrict-classes
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
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDataset
from src.data.preprocessing import build_eval_transform, denormalize
from src.evaluation.gradcam import GradCAM
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import overlay_cam, plot_confusion_matrix, plot_roc_curves
from src.models.model_factory import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config, seed_everything
from src.utils.runs import resolve_run, use_run
from src.utils.tracking import add_log_file, get_logger, log_experiment

logger = get_logger("evaluate", Path("results/logs/evaluate.log"))


@torch.no_grad()
def predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    trues, preds, probs = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        prob = torch.softmax(out, dim=1).cpu().numpy()
        probs.append(prob)
        preds.append(prob.argmax(1))
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds), np.concatenate(probs)


def apply_restrict_classes(y_true, y_pred, y_prob, classes, eval_name):
    """Drop model classes absent from the eval set; returns remapped arrays + classes."""
    present = sorted(set(y_true.tolist()))
    if len(present) >= len(classes):
        return y_true, y_pred, y_prob, classes
    logger.info("Restricting to %d/%d classes present in %s",
                len(present), len(classes), eval_name)
    remap = {c: i for i, c in enumerate(present)}
    y_prob = y_prob[:, present]
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    y_pred = y_prob.argmax(1)
    y_true = np.array([remap[int(t)] for t in y_true])
    return y_true, y_pred, y_prob, [classes[i] for i in present]


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.shape[1] == 1:  # binary edge case
        y_bin = np.hstack([1 - y_bin, y_bin])
    return float(np.mean([
        roc_auc_score(y_bin[:, i], y_prob[:, i]) for i in range(n_classes)]))


def list_checkpoints(model_dir: Path, mode: str) -> list[Path]:
    epoch_ckpts = sorted((model_dir / "epochs").glob("epoch_*.pt"))
    if mode == "epochs":
        return epoch_ckpts
    if mode == "last":
        return [model_dir / "last.pt"]
    return epoch_ckpts + [model_dir / "best.pt", model_dir / "last.pt"]  # all


def plot_ckpt_sweep(df: pd.DataFrame, history_csv: Path, best_epoch: int | None,
                    out_path: Path, title: str, split_label: str = "test") -> None:
    """Eval metrics vs checkpoint epoch, overlaid with the val curve and best.pt mark."""
    ep = df[df["ckpt_file"].str.startswith("epoch")].sort_values("epoch")
    fig, ax = plt.subplots(figsize=(9, 5))
    if not ep.empty:
        ax.plot(ep["epoch"], ep["f1_macro"], "o-", label=f"{split_label} F1", color="tab:blue")
        ax.plot(ep["epoch"], ep["accuracy"], "s--", label=f"{split_label} acc",
                color="tab:blue", alpha=0.5)
    for _, r in df[df["ckpt_file"].isin(("best.pt", "last.pt"))].iterrows():
        ax.scatter(r["epoch"], r["f1_macro"], marker="*" if r["is_best"] else "x",
                   s=140, color="red" if r["is_best"] else "black", zorder=5,
                   label=f'{r["ckpt_file"]} (ep {int(r["epoch"])})')
    if history_csv.exists():
        h = pd.read_csv(history_csv)
        ax.plot(h["epoch"], h["val_f1_macro"], ":", color="tab:green",
                label="val F1 (selection signal)")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="red", ls="--", alpha=0.4,
                   label=f"best.pt = epoch {best_epoch}")
    ax.set_xlabel("Checkpoint epoch"); ax.set_ylabel("Metric")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def sweep_checkpoints(cfg, name: str, ckpts: list[Path], manifest: Path, eval_name: str,
                      max_samples: int | None, device, restrict_classes: bool) -> pd.DataFrame:
    """Evaluate every checkpoint of one model on one split -> sweep table + plot."""
    dataset = BrainTumorDataset(manifest, transform=build_eval_transform(cfg))
    if max_samples:
        dataset.df = dataset.df.head(max_samples)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers)
    model = build_model(cfg, name=name).to(device)
    model_dir = Path(cfg.paths.checkpoints) / name
    best_path = (model_dir / "best.pt").resolve()

    classes = list(cfg.data.classes)
    present = sorted(set(dataset.df["label_idx"].tolist()))
    do_restrict = restrict_classes and len(present) < len(classes)
    eval_classes = [classes[i] for i in present] if do_restrict else classes
    remap = {c: i for i, c in enumerate(present)} if do_restrict else None

    rows = []
    for ck in ckpts:
        ckpt_data = load_checkpoint(ck, model, device=device)
        y_true, y_pred, y_prob = predict_all(model, loader, device)
        if do_restrict:
            y_prob = y_prob[:, present]
            y_prob /= y_prob.sum(axis=1, keepdims=True)
            y_pred = y_prob.argmax(1)
            y_true = np.array([remap[int(t)] for t in y_true])
        m = compute_metrics(y_true, y_pred, eval_classes)
        row = {"model": name, "ckpt_file": ck.name,
               "epoch": int(ckpt_data.get("epoch", -1)),
               "is_best": ck.resolve() == best_path,
               "is_last": ck.name == "last.pt",
               "auc_macro": macro_auc(y_true, y_prob, len(eval_classes)),
               "n_samples": int(len(y_true)), **m}
        rows.append(row)
        logger.info("%s %-14s ep=%2d acc=%.4f f1=%.4f auc=%.4f",
                    name, ck.name, row["epoch"], m["accuracy"], m["f1_macro"], row["auc_macro"])

    df = pd.DataFrame(rows).sort_values(["epoch", "ckpt_file"]).reset_index(drop=True)
    out_metrics = Path(cfg.paths.metrics); out_figs = Path(cfg.paths.figures)
    out_metrics.mkdir(parents=True, exist_ok=True); out_figs.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_metrics / f"{name}_ckpt_sweep_{eval_name}.csv", index=False)

    best_epoch = None
    if best_path.exists():
        b = df[df["ckpt_file"] == "best.pt"]
        if not b.empty:
            best_epoch = int(b.iloc[0]["epoch"])
    plot_ckpt_sweep(df, model_dir / "history.csv", best_epoch,
                    out_figs / f"{name}_ckpt_sweep_{eval_name}.png",
                    title=f"{name} - {eval_name} per-checkpoint metrics",
                    split_label=eval_name)
    return df


def plot_sweep_all(sweeps: dict[str, pd.DataFrame], out_path: Path, eval_name: str) -> None:
    """One line per model: eval F1 vs checkpoint epoch."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in sweeps.items():
        ep = df[df["ckpt_file"].str.startswith("epoch")].sort_values("epoch")
        src = ep if not ep.empty else df
        ax.plot(src["epoch"], src["f1_macro"], "o-", ms=3, label=name)
    ax.set_xlabel("Checkpoint epoch"); ax.set_ylabel(f"{eval_name} F1 (macro)")
    ax.set_title("Per-checkpoint test metrics - all models")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def gradcam_grid(model, dataset, label_names, device, out_path, n=8) -> None:
    cam = GradCAM(model)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    for col, idx in enumerate(indices):
        img_t, label = dataset[idx]
        cam_map = cam.generate(img_t.unsqueeze(0).to(device))
        img = denormalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        axes[0, col].imshow(img); axes[0, col].axis("off")
        axes[0, col].set_title(label_names[label], fontsize=8)
        axes[1, col].imshow(overlay_cam(img, cam_map)); axes[1, col].axis("off")
    fig.suptitle("Grad-CAM")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    cam.close()


def evaluate_one(cfg, name: str, ckpt: Path, manifest: Path, eval_name: str,
                 max_samples: int | None, device, skip_gradcam: bool,
                 restrict_classes: bool, run_id: str = "",
                 by_subject: bool = False) -> dict:
    logger.info("Evaluating %s from %s on %s (%s)", name, ckpt, manifest, device)
    ext = "" if eval_name in ("train", "val", "test") else f"_{eval_name}"

    dataset = BrainTumorDataset(manifest, transform=build_eval_transform(cfg))
    if max_samples:
        dataset.df = dataset.df.head(max_samples)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers)

    model = build_model(cfg, name=name).to(device)
    ckpt_data = load_checkpoint(ckpt, model, device=device)

    y_true, y_pred, y_prob = predict_all(model, loader, device)
    classes = list(cfg.data.classes)
    if restrict_classes:
        y_true, y_pred, y_prob, classes = apply_restrict_classes(
            y_true, y_pred, y_prob, classes, eval_name)
    metrics = compute_metrics(y_true, y_pred, classes)

    out_metrics = Path(cfg.paths.metrics); out_figs = Path(cfg.paths.figures)
    out_metrics.mkdir(parents=True, exist_ok=True); out_figs.mkdir(parents=True, exist_ok=True)

    aucs = plot_roc_curves(y_true, y_prob, classes, out_figs / f"{name}{ext}_roc.png",
                           title=f"{name} - {eval_name} ROC")
    metrics["auc_macro"] = aucs["macro"]
    logger.info("AUC macro: %.4f", aucs["macro"])

    with open(out_metrics / f"{name}_{eval_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics: acc=%.4f prec=%.4f rec=%.4f spec=%.4f f1=%.4f",
                metrics["accuracy"], metrics["precision_macro"], metrics["recall_macro"],
                metrics["specificity_macro"], metrics["f1_macro"])

    # per-sample predictions (for error analysis + McNemar)
    pred_df = dataset.df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["pred_label"] = [classes[i] for i in y_pred]
    pred_df["correct"] = (y_pred == y_true)
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = y_prob[:, i]
    pred_df.to_csv(out_metrics / f"{name}{ext}_predictions.csv", index=False)

    errors = pred_df[~pred_df["correct"]]
    errors.to_csv(out_metrics / f"{name}{ext}_errors.csv", index=False)
    logger.info("Misclassified: %d / %d (%.1f%%)", len(errors), len(pred_df),
                100 * len(errors) / len(pred_df))

    if by_subject and "subject" in pred_df.columns:
        g = pred_df.groupby("subject")
        s_true = g["y_true"].first().to_numpy()
        s_prob = g[[f"prob_{c}" for c in classes]].mean().to_numpy()
        s_metrics = compute_metrics(s_true, s_prob.argmax(1), classes)
        with open(out_metrics / f"{name}{ext}_subject_metrics.json", "w") as f:
            json.dump(s_metrics, f, indent=2)
        logger.info("Subject-level (n=%d): acc=%.4f f1=%.4f",
                    len(s_true), s_metrics["accuracy"], s_metrics["f1_macro"])

    # figures
    plot_confusion_matrix(y_true, y_pred, classes,
                          out_figs / f"{name}{ext}_confusion_matrix.png",
                          title=f"{name} - {eval_name}")

    if not skip_gradcam:
        gradcam_grid(model, dataset, list(cfg.data.classes), device,
                     out_figs / f"{name}{ext}_gradcam.png")

    # append to comparison table
    comp_path = out_metrics / "model_comparison.csv"
    row = pd.DataFrame([{"model": name, "split": eval_name, **{
        k: v for k, v in metrics.items() if not k.endswith("_support")}}])
    if comp_path.exists():
        old = pd.read_csv(comp_path)
        old = old[~((old["model"] == name) & (old["split"] == eval_name))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(comp_path, index=False)
    logger.info("Updated %s", comp_path)

    log_experiment(ckpt_data.get("config") or dict(cfg), name, metrics, eval_name,
                   source="evaluate", path=Path(cfg.paths.results) / "experiments.csv",
                   run_id=run_id)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Model name, comma-separated list, or 'all' (every models/*/best.pt)")
    p.add_argument("--config", default=None, help="Path to YAML config (default: configs/config.yaml)")
    p.add_argument("--checkpoint", default=None, help="Default: <run>/models/<model>/best.pt")
    p.add_argument("--checkpoints", default="best",
                   choices=["best", "last", "epochs", "all"],
                   help="Which checkpoints to evaluate: best (default, full report) | "
                        "last | epochs (epoch_*.pt only) | all (every saved checkpoint). "
                        "Non-best modes produce a per-checkpoint sweep table + curve plot.")
    p.add_argument("--run", default=None, help="run_id, run name, or path (default: latest run)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--csv", default=None,
                   help="External manifest CSV (overrides --split); eval tag = <parent_dir>_<stem>")
    p.add_argument("--restrict-classes", action="store_true",
                   help="Drop model classes absent from the eval set before metrics")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--skip-gradcam", action="store_true")
    p.add_argument("--by-subject", action="store_true",
                   help="If manifest has a 'subject' column, also write patient-level metrics")
    args = p.parse_args()

    cfg = load_config(args.config)
    run_dir = use_run(cfg, resolve_run(cfg, args.run))
    add_log_file(logger, run_dir / "logs" / "evaluate.log")
    logger.info("Run dir: %s", run_dir)
    seed_everything(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = Path(cfg.paths.checkpoints)

    if args.csv:
        manifest = Path(args.csv)
        eval_name = f"{manifest.parent.name}_{manifest.stem}"
    else:
        manifest = Path(cfg.paths.processed_data) / f"{args.split}.csv"
        eval_name = args.split

    if args.model == "all":
        names = sorted(p.parent.name for p in ckpt_root.glob("*/best.pt"))
        if not names:
            logger.error("No checkpoints found under %s", ckpt_root)
            sys.exit(1)
    else:
        names = [n.strip() for n in args.model.split(",") if n.strip()]

    summary = {}
    sweeps = {}
    for name in names:
        if args.checkpoint or args.checkpoints == "best":
            ckpt = Path(args.checkpoint) if args.checkpoint else ckpt_root / name / "best.pt"
            if not ckpt.exists():
                logger.warning("%s: checkpoint %s not found, skipping", name, ckpt)
                continue
            summary[name] = evaluate_one(cfg, name, ckpt, manifest, eval_name,
                                         args.max_samples, device, args.skip_gradcam,
                                         args.restrict_classes, run_dir.name,
                                         by_subject=args.by_subject)
        else:
            ckpts = [c for c in list_checkpoints(ckpt_root / name, args.checkpoints)
                     if c.exists()]
            if not ckpts:
                logger.warning("%s: no checkpoints under %s, skipping",
                               name, ckpt_root / name)
                continue
            sweeps[name] = sweep_checkpoints(cfg, name, ckpts, manifest, eval_name,
                                             args.max_samples, device, args.restrict_classes)
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if sweeps:
        all_df = pd.concat(sweeps.values(), ignore_index=True)
        all_df.to_csv(Path(cfg.paths.metrics) / f"ckpt_sweep_{eval_name}_all.csv", index=False)
        plot_sweep_all(sweeps, Path(cfg.paths.figures) / f"ckpt_sweep_{eval_name}_all.png",
                       eval_name)
        logger.info("=== Checkpoint sweep summary (%s) ===\n%s", eval_name,
                    all_df.pivot_table(index=["model", "epoch", "ckpt_file"],
                                       values=["f1_macro", "accuracy", "auc_macro"])
                    .round(4).to_string())

    if len(summary) > 1:
        logger.info("=== Summary (%s) ===", eval_name)
        for name, m in sorted(summary.items(), key=lambda kv: kv[1]["f1_macro"], reverse=True):
            logger.info("  %-15s acc=%.4f f1=%.4f auc=%.4f", name,
                        m["accuracy"], m["f1_macro"], m["auc_macro"])


if __name__ == "__main__":
    main()
