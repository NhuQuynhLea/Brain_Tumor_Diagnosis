"""Loss function, optimizer, and LR scheduler builders."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def compute_class_weights(manifest_csv: str | Path, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from a split manifest (handles imbalance)."""
    counts = np.bincount(pd.read_csv(manifest_csv)["label_idx"], minlength=num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(cfg, class_weights: torch.Tensor | None = None) -> nn.Module:
    weight = class_weights if cfg.training.get("use_class_weights") else None
    if weight is not None:
        weight = weight.float()
    return nn.CrossEntropyLoss(weight=weight)


def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.training.get("optimizer", "adamw").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.training.learning_rate,
                                weight_decay=cfg.training.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.training.learning_rate, momentum=0.9,
                               weight_decay=cfg.training.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.training.learning_rate,
                             weight_decay=cfg.training.weight_decay)


def build_scheduler(cfg, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    name = cfg.training.get("scheduler", "cosine").lower()
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                         factor=0.1, patience=3)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
