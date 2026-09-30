"""Train all registered models sequentially (tasks 5.5-5.9).

    python scripts/train_all.py                # all 5 models, config defaults
    python scripts/train_all.py --models resnet50 efficientnetb0
    python scripts/train_all.py --epochs 30 --batch-size 32
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.model_factory import available_models
from src.utils.tracking import get_logger

logger = get_logger("train_all", Path("results/logs/train_all.log"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None, help="Subset of models to train")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--debug-batches", type=int, default=None)
    p.add_argument("--extra", nargs="*", default=[], help="Extra args forwarded to train.py")
    args = p.parse_args()

    models = args.models or available_models()
    py = sys.executable
    train_py = Path(__file__).parent / "train.py"

    results = {}
    for name in models:
        cmd = [py, str(train_py), "--model", name]
        if args.epochs: cmd += ["--epochs", str(args.epochs)]
        if args.batch_size: cmd += ["--batch-size", str(args.batch_size)]
        if args.lr: cmd += ["--lr", str(args.lr)]
        if args.debug_batches: cmd += ["--debug-batches", str(args.debug_batches)]
        cmd += args.extra

        logger.info("=== Training %s ===", name)
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
        results[name] = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"

    logger.info("All runs finished:")
    for name, status in results.items():
        logger.info("  %-15s %s", name, status)
    failed = [n for n, s in results.items() if s != "OK"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
