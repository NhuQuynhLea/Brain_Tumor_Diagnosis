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
from src.utils.tracking import get_logger

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


def gradcam_grid(model, dataset, classes, device, out_path, n=8) -> None:
    cam = GradCAM(model)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    for col, idx in enumerate(indices):
        img_t, label = dataset[idx]
        cam_map = cam.generate(img_t.unsqueeze(0).to(device))
        img = denormalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        axes[0, col].imshow(img); axes[0, col].axis("off")
        axes[0, col].set_title(classes[label], fontsize=8)
        axes[1, col].imshow(overlay_cam(img, cam_map)); axes[1, col].axis("off")
    fig.suptitle("Grad-CAM")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    cam.close()


def evaluate_one(cfg, name: str, ckpt: Path, split: str, max_samples: int | None,
                 device, skip_gradcam: bool) -> dict:
    logger.info("Evaluating %s from %s on %s (%s)", name, ckpt, split, device)

    dataset = BrainTumorDataset(Path(cfg.paths.processed_data) / f"{split}.csv",
                                transform=build_eval_transform(cfg))
    if max_samples:
        dataset.df = dataset.df.head(max_samples)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers)

    model = build_model(cfg, name=name).to(device)
    load_checkpoint(ckpt, model, device=device)

    y_true, y_pred, y_prob = predict_all(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, cfg.data.classes)

    out_metrics = Path(cfg.paths.metrics); out_figs = Path(cfg.paths.figures)
    out_metrics.mkdir(parents=True, exist_ok=True); out_figs.mkdir(parents=True, exist_ok=True)

    aucs = plot_roc_curves(y_true, y_prob, cfg.data.classes, out_figs / f"{name}_roc.png",
                           title=f"{name} - ROC")
    metrics["auc_macro"] = aucs["macro"]
    logger.info("AUC macro: %.4f", aucs["macro"])

    with open(out_metrics / f"{name}_{split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics: acc=%.4f prec=%.4f rec=%.4f spec=%.4f f1=%.4f",
                metrics["accuracy"], metrics["precision_macro"], metrics["recall_macro"],
                metrics["specificity_macro"], metrics["f1_macro"])

    # per-sample predictions (for error analysis + McNemar)
    pred_df = dataset.df.copy()
    pred_df["y_pred"] = y_pred
    pred_df["pred_label"] = [cfg.data.classes[i] for i in y_pred]
    pred_df["correct"] = (y_pred == y_true)
    for i, cls in enumerate(cfg.data.classes):
        pred_df[f"prob_{cls}"] = y_prob[:, i]
    pred_df.to_csv(out_metrics / f"{name}_predictions.csv", index=False)

    errors = pred_df[~pred_df["correct"]]
    errors.to_csv(out_metrics / f"{name}_errors.csv", index=False)
    logger.info("Misclassified: %d / %d (%.1f%%)", len(errors), len(pred_df),
                100 * len(errors) / len(pred_df))

    # figures
    plot_confusion_matrix(y_true, y_pred, cfg.data.classes,
                          out_figs / f"{name}_confusion_matrix.png", title=f"{name} - {split}")

    if not skip_gradcam:
        gradcam_grid(model, dataset, cfg.data.classes, device, out_figs / f"{name}_gradcam.png")

    # append to comparison table
    comp_path = out_metrics / "model_comparison.csv"
    row = pd.DataFrame([{"model": name, "split": split, **{
        k: v for k, v in metrics.items() if not k.endswith("_support")}}])
    if comp_path.exists():
        old = pd.read_csv(comp_path)
        old = old[~((old["model"] == name) & (old["split"] == split))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(comp_path, index=False)
    logger.info("Updated %s", comp_path)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Model name, comma-separated list, or 'all' (every models/*/best.pt)")
    p.add_argument("--checkpoint", default=None, help="Default: models/<model>/best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--skip-gradcam", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    seed_everything(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = Path(cfg.paths.checkpoints)

    if args.model == "all":
        names = sorted(p.parent.name for p in ckpt_root.glob("*/best.pt"))
        if not names:
            logger.error("No checkpoints found under %s", ckpt_root)
            sys.exit(1)
    else:
        names = [n.strip() for n in args.model.split(",") if n.strip()]

    summary = {}
    for name in names:
        ckpt = Path(args.checkpoint) if args.checkpoint else ckpt_root / name / "best.pt"
        if not ckpt.exists():
            logger.warning("%s: checkpoint %s not found, skipping", name, ckpt)
            continue
        summary[name] = evaluate_one(cfg, name, ckpt, args.split,
                                     args.max_samples, device, args.skip_gradcam)
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if len(summary) > 1:
        logger.info("=== Summary (%s) ===", args.split)
        for name, m in sorted(summary.items(), key=lambda kv: kv[1]["f1_macro"], reverse=True):
            logger.info("  %-15s acc=%.4f f1=%.4f auc=%.4f", name,
                        m["accuracy"], m["f1_macro"], m["auc_macro"])


if __name__ == "__main__":
    main()
