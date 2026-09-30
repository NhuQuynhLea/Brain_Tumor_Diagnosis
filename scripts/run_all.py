"""Run the full experiment pipeline in one command (designed for background execution):

    data check -> train all models -> evaluate all -> compare/report

Usage:
    python scripts/run_all.py                          # all 5 models, config defaults
    python scripts/run_all.py --models resnet50 efficientnetb0
    python scripts/run_all.py --epochs 30 --batch-size 64
    python scripts/run_all.py --skip-train             # eval + compare only

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
from src.utils.tracking import get_logger

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
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--skip-train", action="store_true", help="Only run evaluate + compare")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--extra", nargs="*", default=[], help="Extra args forwarded to train.py")
    args = p.parse_args()

    py = sys.executable
    scripts = ROOT / "scripts"
    models = args.models or available_models()
    t_start = time.time()

    if not args.skip_train:
        cmd = [py, str(scripts / "train_all.py"), "--models", *models]
        if args.epochs: cmd += ["--epochs", str(args.epochs)]
        if args.batch_size: cmd += ["--batch-size", str(args.batch_size)]
        if args.lr: cmd += ["--lr", str(args.lr)]
        cmd += args.extra
        if not run_step(cmd, f"TRAIN {models}"):
            sys.exit(1)
    else:
        logger.info("Skipping training (--skip-train).")

    if not run_step([py, str(scripts / "evaluate.py"), "--model", "all",
                     "--split", args.split], "EVALUATE all"):
        sys.exit(1)

    if not run_step([py, str(scripts / "compare_models.py")], "COMPARE"):
        sys.exit(1)

    logger.info("Pipeline complete in %.1f min. See results/metrics + results/figures.",
                (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
