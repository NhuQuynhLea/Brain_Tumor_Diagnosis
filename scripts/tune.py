"""Hyperparameter tuning: random search over lr / weight_decay / dropout.

    python scripts/tune.py --model resnet50 --trials 8 --epochs 5
    python scripts/tune.py --model efficientnetb0 --trials 12 --epochs 8 --finetune
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.data.dataset import build_dataloaders
from src.models.model_factory import build_model
from src.training.losses import build_criterion, build_optimizer, build_scheduler, compute_class_weights
from src.training.trainer import Trainer
from src.utils.config import load_config, seed_everything
from src.utils.runs import create_run, finish_run, use_run
from src.utils.tracking import get_logger

logger = get_logger("tune", Path("results/logs/tune.log"))

SEARCH_SPACE = {
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "dropout": [0.3, 0.5],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5, help="Short epochs per trial")
    p.add_argument("--finetune", action="store_true", help="Unfreeze backbone during tuning")
    args = p.parse_args()

    cfg = load_config()
    seed_everything(cfg.project.seed)
    run_dir = create_run(cfg, "tune", args.model)
    logger.info("Run dir: %s", run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(cfg.project.seed)
    combos = list(itertools.product(*SEARCH_SPACE.values()))
    rng.shuffle(combos)
    trials = [dict(zip(SEARCH_SPACE, c)) for c in combos[: args.trials]]

    loaders = build_dataloaders(cfg)
    class_weights = compute_class_weights(
        Path(cfg.paths.processed_data) / "train.csv", len(cfg.data.classes)).to(device)

    results = []
    for i, params in enumerate(trials):
        run_cfg = load_config()
        use_run(run_cfg, run_dir)
        run_cfg.model.name = args.model
        run_cfg.training.epochs = args.epochs
        run_cfg.training.update(params)
        if args.finetune:
            run_cfg.model.freeze_backbone = False
        run_name = f"{args.model}_tune{i}"

        logger.info("Trial %d/%d: %s", i + 1, len(trials), params)
        model = build_model(run_cfg, name=args.model)
        criterion = build_criterion(run_cfg, class_weights)
        optimizer = build_optimizer(run_cfg, model)
        scheduler = build_scheduler(run_cfg, optimizer)

        trainer = Trainer(model, loaders, criterion, optimizer, scheduler, run_cfg,
                          tracker=None, run_name=run_name)
        trainer.train()   # best val metric = trainer.best

        results.append({**params, "best_val_f1": trainer.best,
                        "epochs_ran": len(trainer.history)})
        logger.info("  -> best %s = %.4f", trainer.monitor, trainer.best)

    out = Path(cfg.paths.metrics) / f"tuning_{args.model}.csv"
    df = pd.DataFrame(results).sort_values("best_val_f1", ascending=False)
    df.to_csv(out, index=False)
    best = df.iloc[0].to_dict()
    with open(Path(cfg.paths.metrics) / f"tuning_{args.model}_best.json", "w") as f:
        json.dump(best, f, indent=2)
    finish_run(run_dir)
    logger.info("Best trial: %s", best)


if __name__ == "__main__":
    main()
