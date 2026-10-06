"""Train a single model.

Examples:
    python scripts/train.py --model custom_cnn
    python scripts/train.py --model resnet50 --epochs 30 --batch-size 32 --lr 0.001
    python scripts/train.py --model resnet50 --resume        # resume from last.pt
    python scripts/train.py --model resnet50 --finetune      # unfreeze backbone
    python scripts/train.py --model custom_cnn --debug-batches 10 --epochs 1
"""

import argparse
import shutil
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
from src.utils.runs import create_run, finish_run, use_run
from src.utils.tracking import ExperimentTracker, add_log_file, get_logger, log_experiment

logger = get_logger("train", Path("results/logs/train.log"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="custom_cnn | vgg16 | resnet50 | efficientnetb0 | mobilenetv2 | "
                        "mpac_resnet | lsnet | *_tiny variants")
    p.add_argument("--config", default=None, help="Path to YAML config (default: configs/config.yaml)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", action="store_true", help="Resume from last.pt (--run-dir's run, else models/<model>)")
    p.add_argument("--name", default=None, help="Label appended to the run id")
    p.add_argument("--run-dir", default=None, help="Existing run dir (set by train_all/run_all)")
    p.add_argument("--finetune", action="store_true", help="Unfreeze backbone for full fine-tuning")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=None, help="Override project.seed from config")
    p.add_argument("--patience", type=int, default=None,
                   help="Override early_stopping_patience from config")
    p.add_argument("--data-dir", default=None,
                   help="Override paths.processed_data (e.g. data/external/<name> for task 7.5)")
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
    if args.seed is not None:
        cfg.project.seed = args.seed
    if args.patience is not None:
        cfg.training.early_stopping_patience = args.patience
    if args.data_dir:
        cfg.paths.processed_data = args.data_dir

    seed_everything(cfg.project.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Model=%s device=%s", cfg.model.name, device)

    models_root = Path(cfg.paths.checkpoints)
    own_run = not args.run_dir
    run_dir = create_run(cfg, "train", args.name) if own_run else use_run(cfg, args.run_dir)
    add_log_file(logger, run_dir / "logs" / f"train_{cfg.model.name}.log")
    logger.info("Run dir: %s", run_dir)

    try:
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
                ckpt_root = Path(cfg.paths.checkpoints) if args.run_dir else models_root
                trainer.resume(ckpt_root / cfg.model.name / "last.pt")
            trainer.train()

        # Final test evaluation on best checkpoint
        from src.evaluation.metrics import compute_metrics
        import numpy as np
        load_checkpoint(Path(cfg.paths.checkpoints) / cfg.model.name / "best.pt", model, device=device)
        test_metrics = trainer.validate("test")
        logger.info("TEST %s | acc %.4f f1 %.4f", cfg.model.name,
                    test_metrics["accuracy"], test_metrics["f1_macro"])
        log_experiment(dict(cfg), cfg.model.name, test_metrics, "test",
                       source="train", path=Path(cfg.paths.results) / "experiments.csv",
                       run_id=run_dir.name)

        # models/<model>/ mirrors the latest run
        dst = models_root / cfg.model.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("best.pt", "last.pt", "history.csv"):
            src = Path(cfg.paths.checkpoints) / cfg.model.name / f
            if src.exists():
                shutil.copy2(src, dst / f)
        (dst / "run_id.txt").write_text(run_dir.name)
    except Exception:
        if own_run:
            finish_run(run_dir, "failed")
        raise
    if own_run:
        finish_run(run_dir)


if __name__ == "__main__":
    main()
