"""Train a single model.

Examples:
    python scripts/train.py --model custom_cnn
    python scripts/train.py --model resnet50 --epochs 30 --batch-size 32 --lr 0.001
    python scripts/train.py --model resnet50 --resume        # resume from last.pt
    python scripts/train.py --model resnet50 --finetune      # unfreeze backbone
    python scripts/train.py --model custom_cnn --debug-batches 10 --epochs 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import build_dataloaders
from src.models.model_factory import build_model
from src.training.checkpoint import load_checkpoint
from src.training.losses import build_criterion, build_optimizer, build_scheduler, compute_class_weights
from src.training.trainer import Trainer
from src.utils.config import load_config, seed_everything
from src.utils.tracking import ExperimentTracker, get_logger

logger = get_logger("train", Path("results/logs/train.log"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="custom_cnn | vgg16 | resnet50 | efficientnetb0 | mobilenetv2")
    p.add_argument("--config", default=None, help="Path to YAML config (default: configs/config.yaml)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", action="store_true", help="Resume from models/<model>/last.pt")
    p.add_argument("--finetune", action="store_true", help="Unfreeze backbone for full fine-tuning")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--debug-batches", type=int, default=None,
                   help="Limit batches per epoch (quick pipeline test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    cfg.model.name = args.model
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.finetune:
        cfg.model.freeze_backbone = False
    if args.no_pretrained:
        cfg.model.pretrained = False
    if args.debug_batches:
        cfg.training.debug_batches = args.debug_batches

    seed_everything(cfg.project.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Model=%s device=%s", cfg.model.name, device)

    loaders = build_dataloaders(cfg)
    model = build_model(cfg)

    class_weights = compute_class_weights(
        Path(cfg.paths.processed_data) / "train.csv", len(cfg.data.classes))
    criterion = build_criterion(cfg, class_weights.to(device))
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    with ExperimentTracker(cfg.model.name, Path(cfg.tracking.log_dir)) as tracker:
        trainer = Trainer(model, loaders, criterion, optimizer, scheduler, cfg,
                          tracker=tracker, run_name=cfg.model.name)
        if args.resume:
            trainer.resume(Path(cfg.paths.checkpoints) / cfg.model.name / "last.pt")
        trainer.train()

    # Final test evaluation on best checkpoint
    from src.evaluation.metrics import compute_metrics
    import numpy as np
    load_checkpoint(Path(cfg.paths.checkpoints) / cfg.model.name / "best.pt", model, device=device)
    test_metrics = trainer.validate("test")
    logger.info("TEST %s | acc %.4f f1 %.4f", cfg.model.name,
                test_metrics["accuracy"], test_metrics["f1_macro"])


if __name__ == "__main__":
    main()
