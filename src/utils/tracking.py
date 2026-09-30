"""Experiment tracking: TensorBoard writer + CSV metrics log + console logging."""

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class ExperimentTracker:
    """Thin wrapper around TensorBoard SummaryWriter + CSV metric logging.

    Usage:
        tracker = ExperimentTracker("resnet50", log_dir=Path("results/logs"))
        tracker.log_params({"lr": 1e-3})
        tracker.log_metrics({"train_loss": 0.5}, step=0)
        tracker.close()
    """

    def __init__(self, run_name: str, log_dir: Path, csv_path: Path | None = None):
        self.run_dir = Path(log_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        self.csv_path = csv_path or (self.run_dir / "metrics.csv")
        self._csv_fieldnames: list[str] = []

    def log_params(self, params: dict[str, Any]) -> None:
        self.writer.add_text("hparams", "\n".join(f"- **{k}**: `{v}`" for k, v in params.items()))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        row = {"step": step, **metrics}
        new_keys = [k for k in row if k not in self._csv_fieldnames]
        if new_keys:
            self._csv_fieldnames += new_keys
            self._rewrite_csv_header()
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _rewrite_csv_header(self) -> None:
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def log_figure(self, tag: str, figure, step: int = 0) -> None:
        self.writer.add_figure(tag, figure, global_step=step)

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


EXPERIMENT_COLUMNS = [
    "timestamp", "run_id", "model", "source", "split",
    "pretrained", "freeze_backbone", "dropout",
    "epochs", "batch_size", "lr", "optimizer", "scheduler",
    "weight_decay", "early_stopping_patience", "seed",
    "accuracy", "precision_macro", "recall_macro",
    "specificity_macro", "f1_macro", "auc_macro",
]

_METRIC_KEYS = ("accuracy", "precision_macro", "recall_macro",
                "specificity_macro", "f1_macro", "auc_macro")


def log_experiment(config: dict, model: str, metrics: dict, split: str,
                   source: str, path: Path, run_id: str = "") -> Path:
    """Append one run (config snapshot + metrics) to the experiments CSV table."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "model": model,
        "source": source,
        "split": split,
        "pretrained": model_cfg.get("pretrained"),
        "freeze_backbone": model_cfg.get("freeze_backbone"),
        "dropout": model_cfg.get("dropout"),
        "epochs": train_cfg.get("epochs"),
        "batch_size": train_cfg.get("batch_size"),
        "lr": train_cfg.get("learning_rate"),
        "optimizer": train_cfg.get("optimizer"),
        "scheduler": train_cfg.get("scheduler"),
        "weight_decay": train_cfg.get("weight_decay"),
        "early_stopping_patience": train_cfg.get("early_stopping_patience"),
        "seed": config.get("project", {}).get("seed"),
    }
    for k in _METRIC_KEYS:
        v = metrics.get(k)
        row[k] = round(v, 4) if isinstance(v, float) else v

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            old_header = f.readline()
        if old_header.strip() and "run_id" not in old_header:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                r["run_id"] = ""
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
                w.writeheader()
                w.writerows(rows)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path
