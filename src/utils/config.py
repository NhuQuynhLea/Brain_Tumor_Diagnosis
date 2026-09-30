"""Configuration management: load YAML configs and expose them as attribute-accessible dicts."""

import os
import random
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class Config(dict):
    """Dict with attribute access; nested dicts are wrapped recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config.wrap(v) for v in obj]
        return obj


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load a YAML config file and return a Config object."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config.wrap(raw)

    # Resolve relative paths against project root
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else (PROJECT_ROOT / p)
    return cfg


def seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
