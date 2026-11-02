"""Evaluate a trained model with MCG-TTA (Multi-Scale Consistency-Guided TTA).

--tta accepts: none | full | mcgtta
  none     plain inference - baseline sanity check (must reproduce evaluate.py)
  full     every sample gets K views, prob = mean over views (accuracy ceiling)
  mcgtta   confidence gate (max prob < --tau) -> scale+/- probe views; escalate
           to the remaining views only when the probes disagree with the base
           prediction or with each other. Probes mirror the MPAC block's
           multi-dilation paths: scale-sensitive inputs are exactly where
           multi-path fusion is unstable.

The views pass runs once on the union of gated rows, then the gate/escalation
is replayed exactly per tau -- so --tau-sweep costs one inference pass.

Outputs land in a NEW run dir: results/runs/<ts>_tta_<name>/metrics/
  <model>_<variant>[_<split>]_predictions.csv   (variant e.g. base/tta6/mcgtta6_t80)
  *_metrics.json, *_subject_metrics.json, *_timing.json, confusion matrix + ROC
  tta_strategies_summary.csv when a tau sweep is evaluated

    python scripts/evaluate_tta.py --model lsnet --config configs/config_brats.yaml ^
        --checkpoint results/runs/<run>/models/lsnet/best.pt --tta mcgtta --tau 0.8
    python scripts/evaluate_tta.py --model lsnet --tta mcgtta --split val ^
        --tau-sweep 0.60 0.70 0.75 0.80 0.85 --checkpoint <ckpt>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scripts.evaluate import apply_restrict_classes, macro_auc
from src.evaluation.complexity import measure_inference_time
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curves
from src.evaluation.tta import (build_tta_views, forward_views, load_raw_images,
                                run_tta_eval, simulate_mcgtta, union_trigger)
from src.models.model_factory import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config, seed_everything
from src.utils.runs import create_run, finish_run
from src.utils.tracking import add_log_file, get_logger, log_experiment

logger = get_logger("evaluate_tta", Path("results/logs/evaluate_tta.log"))

MODES = ("none", "full", "mcgtta")


def parse_modes(spec: str) -> list[str]:
    out = []
    for s in spec.split(","):
        s = s.strip().lower()
        if s in ("mcgtta", "scalempac", "all", "selective"):  # legacy aliases
            out.append("mcgtta")
        elif s in MODES:
            out.append(s)
        else:
            raise SystemExit(f"unknown --tta '{s}'; choices: {','.join(MODES)}")
    return list(dict.fromkeys(out))


def variant_name(mode: str, tau: float, n_views: int) -> str:
    if mode == "none":
        return "base"
    if mode == "full":
        return f"tta{n_views}"
    return f"{mode}{n_views}_t{int(round(tau * 100))}"


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


def log_line(stem: str, m: dict, subj: dict | None,
             trigger_rate: float | None = None) -> None:
    trig = f" trig={trigger_rate:.1%}" if trigger_rate is not None else ""
    s = (f" subj[acc={subj['accuracy']:.4f} f1={subj['f1_macro']:.4f}]"
         if subj else "")
    logger.info("%-44s acc=%.4f f1=%.4f auc=%.4f%s%s", stem,
                m["accuracy"], m["f1_macro"], m["auc_macro"], trig, s)


def summary_row(model_name, mode, variant, tau, n_samples, n_trig, n_esc,
                extra, base_single_ms, m, subj, classes):
    row = {"model": model_name, "strategy": mode, "variant": variant,
           "tau": tau, "n_triggered": int(n_trig),
           "trigger_rate": round(n_trig / n_samples, 4),
           "n_escalated": int(n_esc), "extra_forwards": int(extra),
           "modelled_ms": round(base_single_ms * (1 + extra / n_samples), 2),
           "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
           "auc_macro": m["auc_macro"]}
    if subj:
        row["subject_acc"] = subj["accuracy"]
        row["subject_f1"] = subj["f1_macro"]
        for c in classes:
            row[f"{c}_caught"] = subj.get(f"{c}_caught")
            row[f"{c}_support"] = subj.get(f"{c}_support")
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint (e.g. results/runs/<run>/models/<model>/best.pt)")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--csv", default=None, help="Manifest CSV (overrides --split)")
    p.add_argument("--tta", default="mcgtta",
                   help="none|full|mcgtta (comma-separated list allowed)")
    p.add_argument("--tau", type=float, default=0.8,
                   help="Confidence gate: enter MCG-TTA when max prob < tau")
    p.add_argument("--tau-sweep", nargs="+", type=float, default=None,
                   help="Evaluate a grid of taus in one pass (views computed once)")
    p.add_argument("--tta-views", type=int, default=6,
                   help="Total views including identity (max 6; mcgtta uses the scale views)")
    p.add_argument("--budget", type=float, default=0.28,
                   help="Max trigger rate used for the sweep's tau recommendation")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Stratified subsample for smoke tests")
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
    if args.max_samples and len(df) > args.max_samples:
        # stratified subsample (keeps class ratio + subject diversity); head()
        # would pick ~4 subjects of a single class and break subject metrics
        parts = []
        for _, g in df.groupby("label_idx"):
            n = max(1, round(len(g) * args.max_samples / len(df)))
            parts.append(g.sample(n=n, random_state=cfg.project.seed))
        df = pd.concat(parts).sort_index()
    df = df.reset_index(drop=True)
    logger.info("Manifest %s -> %d samples (%s)", manifest, len(df), eval_name)

    modes = parse_modes(args.tta)
    logger.info("Modes: %s | device: %s", modes, device)

    model = build_model(cfg, name=args.model).to(device)
    ckpt_data = load_checkpoint(Path(args.checkpoint), model, device=device)
    logger.info("Checkpoint: %s (epoch %s)", args.checkpoint, ckpt_data.get("epoch"))

    # Single-image latency on the same protocol as profile_models.py, so the
    # modelled effective latency is comparable to the paper's latency column.
    base_single_ms = measure_inference_time(
        model, (1, 3, cfg.data.image_size, cfg.data.image_size), device)
    logger.info("Base single-image latency: %.2f ms", base_single_ms)

    classes = list(cfg.data.classes)
    y_true = df["label_idx"].to_numpy()

    try:
        # Real wall-clock path for none/full (single mode, no sweep)
        if len(modes) == 1 and modes[0] in ("none", "full") and not args.tau_sweep:
            res = run_tta_eval(model, df, cfg, device, mode=modes[0],
                               n_views=args.tta_views, batch_size=batch_size)
            variant = variant_name(modes[0], args.tau, args.tta_views)
            probs, trigger = res["probs"], res["trigger"]
            yt = y_true
            if args.restrict_classes:
                yt, _, probs, classes = apply_restrict_classes(
                    y_true, probs.argmax(1), probs, classes, eval_name)
            m, subj = save_outputs(cfg, args.model, variant, eval_name, df, yt,
                                   probs, classes, trigger, res["conf"])
            k = len(res["views"])
            timing = {**res["timing"],
                      "base_single_ms": base_single_ms,
                      "modelled_ms": round(
                          base_single_ms * (1 + trigger.mean() * (k - 1)), 2)}
            ext = "" if eval_name == "test" else f"_{eval_name}"
            stem = f"{args.model}_{variant}{ext}"
            with open(Path(cfg.paths.metrics) / f"{stem}_timing.json", "w") as f:
                json.dump(timing, f, indent=2)
            log_line(stem, m, subj, trigger.mean())
            logger.info("Timing: %s | modelled single-image %.2f ms",
                        json.dumps(timing), timing["modelled_ms"])
            log_experiment(ckpt_data.get("config") or dict(cfg),
                           f"{args.model}_{variant}", m, eval_name,
                           source="evaluate_tta",
                           path=Path(cfg.paths.results) / "experiments.csv",
                           run_id=run_dir.name)

        # Union pass + exact gate/escalation replay (mcgtta, sweeps, mixed lists)
        else:
            taus = args.tau_sweep or [args.tau]
            views = build_tta_views(cfg)[:max(1, args.tta_views)]
            view_names = [v[0] for v in views]
            n = len(df)
            if "mcgtta" in modes and not any(v.startswith("scale") for v in view_names):
                logger.warning("mcgtta needs scale probe views; use --tta-views 5 or 6 "
                               "(got %d)", len(views))

            imgs = load_raw_images(df)
            p0 = forward_views(model, imgs, views[:1], device, batch_size,
                               desc="Pass 1 (plain)")[:, 0]
            if device.type == "cuda":
                torch.cuda.synchronize()

            idx_T = np.arange(n) if "full" in modes else union_trigger(p0, taus)
            logger.info("Union trigger set: %d/%d rows (%.1f%%) get extra views once",
                        len(idx_T), n, 100 * len(idx_T) / n)
            rest = forward_views(model, [imgs[i] for i in idx_T], views[1:],
                                 device, batch_size,
                                 desc=f"Views on {len(idx_T)} gated rows") \
                if len(idx_T) and len(views) > 1 \
                else np.zeros((len(idx_T), 0, p0.shape[1]))
            probs_T = np.concatenate([p0[idx_T][:, None], rest], axis=1)

            rows = []
            combos = [(t, s) for t in taus for s in modes]
            for tau, mode in tqdm(combos, desc="Evaluating"):
                if mode == "none":
                    probs, trig = p0.copy(), np.zeros(n, dtype=bool)
                    extra, n_esc = 0, 0
                elif mode == "full":
                    # idx_T covers all rows, so probs_T has every view
                    probs = probs_T.mean(axis=1)
                    trig = np.ones(n, dtype=bool)
                    extra, n_esc = n * (len(views) - 1), 0
                else:
                    probs, trig, extra, n_esc = simulate_mcgtta(
                        probs_T, idx_T, p0, tau, view_names)
                variant = variant_name(mode, tau, args.tta_views)
                yt = y_true
                pc = probs
                if args.restrict_classes:
                    yt, _, pc, classes = apply_restrict_classes(
                        y_true, pc.argmax(1), pc, classes, eval_name)
                m, subj = save_outputs(cfg, args.model, variant, eval_name, df,
                                       yt, pc, classes, trig, p0.max(1),
                                       figures=(len(combos) == 1))
                ext = "" if eval_name == "test" else f"_{eval_name}"
                stem = f"{args.model}_{variant}{ext}"
                timing = {"mode": mode, "tau": tau, "n_views": len(views),
                          "n_samples": n, "n_triggered": int(trig.sum()),
                          "trigger_rate": float(trig.mean()),
                          "n_escalated": n_esc, "extra_forwards": extra,
                          "base_single_ms": base_single_ms,
                          "modelled_ms": round(
                              base_single_ms * (1 + extra / n), 2)}
                with open(Path(cfg.paths.metrics) / f"{stem}_timing.json", "w") as f:
                    json.dump(timing, f, indent=2)
                log_line(stem, m, subj, trig.mean())
                logger.info("  %s: gated=%d escalated=%d extra_fwd=%d "
                            "modelled=%.2f ms", stem, int(trig.sum()), n_esc,
                            extra, timing["modelled_ms"])
                rows.append(summary_row(args.model, mode, variant, tau, n,
                                        int(trig.sum()), n_esc, extra,
                                        base_single_ms, m, subj, classes))
                if len(taus) == 1:
                    log_experiment(ckpt_data.get("config") or dict(cfg),
                                   f"{args.model}_{variant}", m, eval_name,
                                   source="evaluate_tta",
                                   path=Path(cfg.paths.results) / "experiments.csv",
                                   run_id=run_dir.name)

            out = Path(cfg.paths.metrics) / "tta_strategies_summary.csv"
            summ = pd.DataFrame(rows)
            summ.to_csv(out, index=False)
            logger.info("Wrote %s\n%s", out, summ.round(4).to_string(index=False))
            if args.tau_sweep:
                ok = summ[summ["trigger_rate"] <= args.budget]
                if not ok.empty:
                    best = ok.loc[ok["f1_macro"].idxmax()]
                    logger.info("Recommended: %s tau=%.2f (trigger %.1f%%, f1=%.4f)",
                                best["strategy"], best["tau"],
                                best["trigger_rate"] * 100, best["f1_macro"])
        finish_run(run_dir, "completed")
    except Exception:
        finish_run(run_dir, "failed")
        raise


if __name__ == "__main__":
    main()
