"""Training loop: train/validate epochs, early stopping, checkpointing, tracking."""

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.utils.tracking import ExperimentTracker, get_logger


class Trainer:
    """Full training pipeline for one model run.

    Monitors cfg.training.monitor (default 'val_f1_macro', mode max) for
    best-checkpoint selection and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: dict,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        cfg,
        tracker: ExperimentTracker | None = None,
        run_name: str | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tracker = tracker
        self.classes = list(cfg.data.classes)
        self.run_name = run_name or cfg.model.name
        self.logger = get_logger(f"trainer.{self.run_name}",
                                 Path(cfg.paths.logs) / f"train_{self.run_name}.log")
        self.ckpt_dir = Path(cfg.paths.checkpoints) / self.run_name
        self.epochs_dir = self.ckpt_dir / "epochs"
        self.save_epochs = bool(cfg.training.get("save_epoch_checkpoints", True))
        self.max_batches = int(cfg.training.get("debug_batches", 0)) or None

        monitor = cfg.training.get("monitor", "val_f1_macro")
        self.monitor = monitor
        self.monitor_mode = "min" if "loss" in monitor else "max"
        self.best = np.inf if self.monitor_mode == "min" else -np.inf
        self.epochs_no_improve = 0
        self.start_epoch = 0
        self.history: list[dict] = []

    # ---- loops ----

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        losses, preds, trues = [], [], []
        loader = self.loaders["train"]
        n = self.max_batches or len(loader)
        pbar = tqdm(loader, total=n, desc=f"[{self.run_name}] epoch {epoch} train", leave=False)
        for i, (x, y) in enumerate(pbar):
            if self.max_batches and i >= self.max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            clip = self.cfg.training.get("grad_clip", 0)
            if clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], clip)
            self.optimizer.step()

            losses.append(loss.item())
            preds.extend(out.argmax(1).detach().cpu().numpy())
            trues.extend(y.cpu().numpy())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
        m = compute_metrics(np.array(trues), np.array(preds), self.classes)
        return {"loss": float(np.mean(losses)), "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}

    @torch.no_grad()
    def validate(self, split: str = "val") -> dict:
        self.model.eval()
        losses, preds, trues = [], [], []
        loader = self.loaders[split]
        n = self.max_batches or len(loader)
        for i, (x, y) in enumerate(tqdm(loader, total=n, desc=f"[{self.run_name}] {split}", leave=False)):
            if self.max_batches and i >= self.max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            losses.append(self.criterion(out, y).item())
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y.cpu().numpy())
        m = compute_metrics(np.array(trues), np.array(preds), self.classes)
        return {"loss": float(np.mean(losses)), **m}

    # ---- checkpoint ----

    def _is_better(self, value: float) -> bool:
        return value < self.best if self.monitor_mode == "min" else value > self.best

    def resume(self, path: str | Path) -> None:
        ckpt = load_checkpoint(path, self.model, self.optimizer, self.scheduler, self.device)
        self.start_epoch = ckpt["epoch"] + 1
        self.best = ckpt["metrics"].get(self.monitor, self.best)
        hist_path = self.ckpt_dir / "history.csv"
        if hist_path.exists():
            self.history = pd.read_csv(hist_path).to_dict("records")
        self.logger.info("Resumed from %s at epoch %d", path, self.start_epoch)

    # ---- main ----

    def train(self) -> pd.DataFrame:
        epochs = int(self.cfg.training.epochs)
        patience = int(self.cfg.training.early_stopping_patience)
        self.logger.info("Training %s on %s for up to %d epochs (monitor=%s, patience=%d)",
                         self.run_name, self.device, epochs, self.monitor, patience)
        if self.tracker:
            self.tracker.log_params({**dict(self.cfg.training), **dict(self.cfg.model),
                                     "model": self.run_name, "device": str(self.device)})

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            tr = self.train_epoch(epoch)
            va = self.validate("val")
            row = {
                "epoch": epoch,
                "train_loss": tr["loss"], "train_accuracy": tr["accuracy"], "train_f1_macro": tr["f1_macro"],
                "val_loss": va["loss"], "val_accuracy": va["accuracy"], "val_f1_macro": va["f1_macro"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_sec": round(time.time() - t0, 1),
            }
            self.history.append(row)
            if self.tracker:
                self.tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            self.logger.info(
                "epoch %d | train loss %.4f acc %.4f | val loss %.4f acc %.4f f1 %.4f | %.0fs",
                epoch, tr["loss"], tr["accuracy"], va["loss"], va["accuracy"],
                va["f1_macro"], row["epoch_sec"])

            # scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(va["loss"])
            elif self.scheduler is not None:
                self.scheduler.step()

            # checkpoints
            state = {**row, **{f"val_{k}": v for k, v in va.items() if not k.startswith("val_")}}
            save_checkpoint(self.ckpt_dir / "last.pt", self.model, self.optimizer,
                            self.scheduler, epoch, state, dict(self.cfg))
            if self.save_epochs:
                save_checkpoint(self.epochs_dir / f"epoch_{epoch:03d}.pt", self.model,
                                None, None, epoch, state, dict(self.cfg))
            self._append_history_row(row)
            if self._is_better(va[self.monitor.removeprefix("val_")]):
                self.best = va[self.monitor.removeprefix("val_")]
                self.epochs_no_improve = 0
                save_checkpoint(self.ckpt_dir / "best.pt", self.model, self.optimizer,
                                self.scheduler, epoch, state, dict(self.cfg))
                self.logger.info("  -> new best %s=%.4f, saved best.pt", self.monitor, self.best)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= patience:
                    self.logger.info("Early stopping at epoch %d.", epoch)
                    break

        history = pd.DataFrame(self.history)
        history.to_csv(self.ckpt_dir / "history.csv", index=False)
        self.logger.info("Done. Best %s=%.4f", self.monitor, self.best)
        return history

    def _append_history_row(self, row: dict) -> None:
        """Append one epoch row to history.csv so it survives a crash."""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / "history.csv"
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
