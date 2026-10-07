"""Run the full experiment pipeline in one command (designed for background execution):

    data check -> train all models -> evaluate all -> compare/report

Usage:
    python scripts/run_all.py                          # all 5 models, config defaults
    python scripts/run_all.py --models resnet50 efficientnetb0
    python scripts/run_all.py --epochs 30 --batch-size 64
    python scripts/run_all.py --skip-train             # eval + compare only
    python scripts/run_all.py --models resnet50 efficientnetb0 mobilenetv2 --epochs 30 --lr 0.0001 --extra --finetune

Background on the GPU machine:
    nohup python scripts/run_all.py > results/logs/run_all.out 2>&1 &
    tail -f results/logs/run_all.out

All step output is also appended to results/logs/run_all.log.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.model_factory import available_models
from src.utils.config import load_config
from src.utils.runs import create_run, finish_run, resolve_run
from src.utils.tracking import add_log_file, get_logger

ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("run_all", ROOT / "results" / "logs" / "run_all.log")


def run_step(cmd: list[str], label: str) -> bool:
    logger.info("=== %s ===\n$ %s", label, " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    ok = proc.returncode == 0
    logger.info("%s %s in %.0fs", label, "OK" if ok else f"FAILED (rc={proc.returncode})",
                time.time() - t0)
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None, help="Subset of models")
    p.add_argument("--config", default=None, help="Path to YAML config (default: configs/config.yaml)")
    p.add_argument("--seed", type=int, default=None, help="Override project.seed (multi-seed runs)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--skip-train", action="store_true", help="Only run evaluate + compare")
    p.add_argument("--run", default=None, help="With --skip-train: run to evaluate/compare")
    p.add_argument("--name", default=None, help="Label appended to the run id")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--by-subject", action="store_true",
                   help="Forward to evaluate.py: also write patient-level metrics")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is forwarded to train.py (must be last)")
    args = p.parse_args()

    py = sys.executable
    scripts = ROOT / "scripts"
    models = args.models or available_models()
    t_start = time.time()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.project.seed = args.seed

    if args.skip_train:
        run_dir = resolve_run(cfg, args.run)
        add_log_file(logger, run_dir / "logs" / "run_all.log")
        logger.info("Skipping training; using run %s", run_dir.name)
        own_run = False
    else:
        run_dir = create_run(cfg, "run_all", args.name)
        add_log_file(logger, run_dir / "logs" / "run_all.log")
        own_run = True
        cmd = [py, str(scripts / "train_all.py"), "--models", *models,
               "--run-dir", str(run_dir)]
        if args.epochs: cmd += ["--epochs", str(args.epochs)]
        if args.batch_size: cmd += ["--batch-size", str(args.batch_size)]
        if args.lr: cmd += ["--lr", str(args.lr)]
        if args.config: cmd += ["--config", args.config]
        if args.seed is not None: cmd += ["--seed", str(args.seed)]
        if args.extra: cmd += ["--extra", *args.extra]
        if not run_step(cmd, f"TRAIN {models}"):
            finish_run(run_dir, "failed")
            sys.exit(1)

    eval_cmd = [py, str(scripts / "evaluate.py"), "--model", "all",
                "--split", args.split, "--run", str(run_dir)]
    cmp_cmd = [py, str(scripts / "compare_models.py"), "--run", str(run_dir)]
    if args.config:
        eval_cmd += ["--config", args.config]
        cmp_cmd += ["--config", args.config]
    if args.by_subject:
        eval_cmd += ["--by-subject"]

    if not run_step(eval_cmd, "EVALUATE all"):
        if own_run: finish_run(run_dir, "failed")
        sys.exit(1)

    if not run_step(cmp_cmd, "COMPARE"):
        if own_run: finish_run(run_dir, "failed")
        sys.exit(1)

    if own_run:
        finish_run(run_dir)
    logger.info("Pipeline complete in %.1f min. Run: %s",
                (time.time() - t_start) / 60, run_dir)


if __name__ == "__main__":
    main()
