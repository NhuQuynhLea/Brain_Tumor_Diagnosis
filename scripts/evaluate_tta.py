"""Evaluate a trained model with test-time augmentation (TTA).

Modes (--tta):
  none       plain inference - baseline sanity check (must reproduce evaluate.py)
  full       every sample gets K views, prob = mean over views (accuracy ceiling)
  selective  only low-confidence samples (max prob < --tau) get extra views
  subject    gate at patient level: all slices of a low-confidence subject get TTA

Outputs land in a NEW run dir: results/runs/<ts>_tta_<name>/metrics/
  <model>_<variant>[_<split>]_predictions.csv   (variant e.g. base/tta5/stta5_t80)
  *_metrics.json, *_subject_metrics.json, *_timing.json, confusion matrix + ROC
  --tau-sweep additionally writes tta_sweep_summary.csv (one row per tau)

    python scripts/evaluate_tta.py --model lsnet --config configs/config_brats.yaml ^
        --checkpoint results/runs/20261007_221751_run_all_brats_ft_s42/models/lsnet/best.pt ^
        --tta selective --tau 0.8 --split test --name lsnet_stta80_s42
    # tau sweep on val (single all-views pass, gating simulated per tau):
    python scripts/evaluate_tta.py --model lsnet --config configs/config_brats.yaml ^
        --checkpoint <ckpt> --tta selective --split val ^
        --tau-sweep 0.60 0.70 0.75 0.80 0.85 --name lsnet_sweep_s42
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from scripts.evaluate import apply_restrict_classes, macro_auc
from src.evaluation.complexity import measure_inference_time
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curves
from src.evaluation.tta import (compute_view_probs, gate_and_aggregate,
                                run_tta_eval)
from src.models.model_factory import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config, seed_everything
from src.utils.runs import create_run, finish_run
from src.utils.tracking import add_log_file, get_logger, log_experiment

logger = get_logger("evaluate_tta", Path("results/logs/evaluate_tta.log"))


def variant_name(mode: str, tau: float, n_views: int) -> str:
    if mode == "none":
        return "base"
    if mode == "full":
        return f"tta{n_views}"
    tag = "subjtta" if mode == "subject" else "stta"
    return f"{tag}{n_views}_t{int(round(tau * 100))}"


def subject_stats(df: pd.DataFrame, probs: np.ndarray, classes: list[str]) -> dict | None:
    """Patient-level metrics: mean slice probs per subject -> argmax (same as evaluate.py)."""
    if "subject" not in df.columns:
        return None
    tmp = df.copy()
    for i, c in enumerate(classes):
        tmp[f"prob_{c}"] = probs[:, i]
    g = tmp.groupby("subject")
    s_true = g["label_idx"].first().to_numpy()
    s_prob = g[[f"prob_{c}" for c in classes]].mean().to_numpy()
    m = compute_metrics(s_true, s_prob.argmax(1), classes)
    m["n_subjects"] = int(len(s_true))
    for i, c in enumerate(classes):
        m[f"{c}_caught"] = int(round(m[f"{c}_recall"] * m[f"{c}_support"]))
    return m


def save_outputs(cfg, model_name: str, variant: str, eval_name: str,
                 df: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray,
                 classes: list[str], trigger: np.ndarray | None = None,
                 conf: np.ndarray | None = None, figures: bool = True) -> tuple[dict, dict | None]:
    """Write metrics/predictions/errors/subject metrics; return (slice_metrics, subj_metrics)."""
    ext = "" if eval_name == "test" else f"_{eval_name}"
    out_metrics = Path(cfg.paths.metrics)
    out_figs = Path(cfg.paths.figures)
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_{variant}{ext}"

    y_pred = probs.argmax(1)
    metrics = compute_metrics(y_true, y_pred, classes)
    if figures:
        aucs = plot_roc_curves(y_true, probs, classes, out_figs / f"{stem}_roc.png",
                               title=f"{stem} ROC")
        metrics["auc_macro"] = aucs["macro"]
    else:
        metrics["auc_macro"] = macro_auc(y_true, probs, len(classes))

    with open(out_metrics / f"{stem}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["pred_label"] = [classes[i] for i in y_pred]
    pred_df["correct"] = (y_pred == y_true)
    if conf is not None:
        pred_df["conf_base"] = conf
    if trigger is not None:
        pred_df["tta_triggered"] = trigger
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = probs[:, i]
    pred_df.to_csv(out_metrics / f"{stem}_predictions.csv", index=False)
    pred_df[~pred_df["correct"]].to_csv(out_metrics / f"{stem}_errors.csv", index=False)

    subj = subject_stats(pred_df, probs, classes)
    if subj is not None:
        with open(out_metrics / f"{stem}_subject_metrics.json", "w") as f:
            json.dump(subj, f, indent=2)

    if figures:
        plot_confusion_matrix(y_true, y_pred, classes,
                              out_figs / f"{stem}_confusion_matrix.png", title=stem)
    return metrics, subj


def log_line(eval_name: str, stem: str, m: dict, subj: dict | None,
             trigger_rate: float | None = None) -> None:
    trig = f" trig={trigger_rate:.1%}" if trigger_rate is not None else ""
    s = (f" subj[acc={subj['accuracy']:.4f} f1={subj['f1_macro']:.4f}]"
         if subj else "")
    logger.info("%-44s acc=%.4f f1=%.4f auc=%.4f%s%s", stem,
                m["accuracy"], m["f1_macro"], m["auc_macro"], trig, s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint (e.g. results/runs/<run>/models/<model>/best.pt)")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--csv", default=None, help="Manifest CSV (overrides --split)")
    p.add_argument("--tta", default="selective",
                   choices=["none", "full", "selective", "subject"])
    p.add_argument("--tau", type=float, default=0.8,
                   help="Confidence gate: trigger TTA when max prob < tau")
    p.add_argument("--tau-sweep", nargs="+", type=float, default=None,
                   help="Evaluate a grid of taus in one pass (views computed once)")
    p.add_argument("--tta-views", type=int, default=5,
                   help="Total views including identity (max 6)")
    p.add_argument("--budget", type=float, default=0.28,
                   help="Max trigger rate used for the sweep's tau recommendation")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--restrict-classes", action="store_true")
    p.add_argument("--name", default=None, help="Run name suffix: <ts>_tta_<name>")
    args = p.parse_args()

    cfg = load_config(args.config)
    run_dir = create_run(cfg, "tta", args.name)
    add_log_file(logger, run_dir / "logs" / "evaluate_tta.log")
    logger.info("Run dir: %s", run_dir)
    seed_everything(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or cfg.training.batch_size

    manifest = Path(args.csv) if args.csv else Path(cfg.paths.processed_data) / f"{args.split}.csv"
    eval_name = f"{manifest.parent.name}_{manifest.stem}" if args.csv else args.split
    df = pd.read_csv(manifest)
    if args.max_samples:
        df = df.head(args.max_samples)
    if args.tta == "subject" and "subject" not in df.columns:
        raise SystemExit("--tta subject requires a 'subject' column in the manifest")
    logger.info("Manifest %s -> %d samples (%s)", manifest, len(df), eval_name)

    model = build_model(cfg, name=args.model).to(device)
    ckpt_data = load_checkpoint(Path(args.checkpoint), model, device=device)
    logger.info("Checkpoint: %s (epoch %s)", args.checkpoint, ckpt_data.get("epoch"))

    # Single-image latency on the same protocol as profile_models.py, so the
    # modelled effective latency is comparable to the paper's latency column.
    base_single_ms = measure_inference_time(
        model, (1, 3, cfg.data.image_size, cfg.data.image_size), device)

    classes = list(cfg.data.classes)
    y_true = df["label_idx"].to_numpy()
    if args.restrict_classes:
        _, _, _, classes = apply_restrict_classes(
            y_true, y_true, np.zeros((len(df), len(classes))), classes, eval_name)

    name = args.model
    try:
        if args.tau_sweep:
            if args.tta not in ("selective", "subject"):
                raise SystemExit("--tau-sweep requires --tta selective|subject")
            probs_views, df, elapsed = compute_view_probs(
                model, df, cfg, device, args.tta_views, batch_size)
            ms_per_view = elapsed / (len(df) * probs_views.shape[1]) * 1000
            rows = []
            for tau in args.tau_sweep:
                variant = variant_name(args.tta, tau, args.tta_views)
                probs, trigger = gate_and_aggregate(probs_views, df, tau, args.tta)
                yt = y_true
                if args.restrict_classes:
                    yt, _, probs, classes = apply_restrict_classes(
                        y_true, probs.argmax(1), probs, classes, eval_name)
                m, subj = save_outputs(cfg, name, variant, eval_name, df, yt, probs,
                                       classes, trigger, probs_views[:, 0].max(1),
                                       figures=False)
                r = trigger.mean()
                row = {"model": name, "variant": variant, "tau": tau,
                       "trigger_rate": round(r, 4),
                       "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                       "auc_macro": m["auc_macro"],
                       "modelled_ms": round(base_single_ms * (1 + r * (args.tta_views - 1)), 2),
                       "subject_acc": subj["accuracy"] if subj else None,
                       "subject_f1": subj["f1_macro"] if subj else None}
                for c in classes:
                    if subj:
                        row[f"{c}_caught"] = subj.get(f"{c}_caught")
                        row[f"{c}_support"] = subj.get(f"{c}_support")
                rows.append(row)
                log_line(eval_name, f"{name}_{variant}", m, subj, r)
            sweep = pd.DataFrame(rows)
            out = Path(cfg.paths.metrics) / "tta_sweep_summary.csv"
            sweep.to_csv(out, index=False)
            ok = sweep[sweep["trigger_rate"] <= args.budget]
            best = (ok if not ok.empty else sweep).loc[
                lambda s: s["f1_macro"].idxmax()]
            logger.info("Wrote %s\n%s", out, sweep.round(4).to_string(index=False))
            logger.info("Recommended tau=%.2f (trigger %.1f%%, f1=%.4f; budget %.0f%%)",
                        best["tau"], best["trigger_rate"] * 100, best["f1_macro"],
                        args.budget * 100)
        else:
            res = run_tta_eval(model, df, cfg, device, mode=args.tta, tau=args.tau,
                               n_views=args.tta_views, batch_size=batch_size)
            variant = variant_name(args.tta, args.tau, args.tta_views)
            probs, trigger = res["probs"], res["trigger"]
            yt = y_true
            if args.restrict_classes:
                yt, _, probs, classes = apply_restrict_classes(
                    y_true, probs.argmax(1), probs, classes, eval_name)
            m, subj = save_outputs(cfg, name, variant, eval_name, df, yt, probs,
                                   classes, trigger, res["conf"])
            timing = {**res["timing"],
                      "base_single_ms": base_single_ms,
                      "modelled_ms": round(
                          base_single_ms * (1 + trigger.mean() * (len(res["views"]) - 1)), 2)}
            stem = f"{name}_{variant}{'' if eval_name == 'test' else '_' + eval_name}"
            with open(Path(cfg.paths.metrics) / f"{stem}_timing.json", "w") as f:
                json.dump(timing, f, indent=2)
            log_line(eval_name, stem, m, subj, trigger.mean())
            logger.info("Timing: %s | modelled single-image %.2f ms",
                        json.dumps(timing), timing["modelled_ms"])
            log_experiment(ckpt_data.get("config") or dict(cfg), f"{name}_{variant}",
                           m, eval_name, source="evaluate_tta",
                           path=Path(cfg.paths.results) / "experiments.csv",
                           run_id=run_dir.name)
        finish_run(run_dir, "completed")
    except Exception:
        finish_run(run_dir, "failed")
        raise


if __name__ == "__main__":
    main()
