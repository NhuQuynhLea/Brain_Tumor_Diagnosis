"""PyTorch Dataset and DataLoader construction from split manifests."""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class BrainTumorDataset(Dataset):
    """Reads a manifest CSV with columns [filepath, label, label_idx] and returns (image, label)."""

    def __init__(self, manifest_csv: str | Path, transform: A.Compose | None = None):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = cv2.imread(str(row["filepath"]), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not read image: {row['filepath']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, int(row["label_idx"])

    @property
    def labels(self) -> np.ndarray:
        return self.df["label_idx"].to_numpy()


def build_dataloaders(cfg) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from the split manifests in data/processed."""
    from src.data.augmentation import build_train_transform
    from src.data.preprocessing import build_eval_transform

    processed = Path(cfg.paths.processed_data)
    train_tf = build_train_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    datasets = {
        "train": BrainTumorDataset(processed / "train.csv", transform=train_tf),
        "val": BrainTumorDataset(processed / "val.csv", transform=eval_tf),
        "test": BrainTumorDataset(processed / "test.csv", transform=eval_tf),
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.training.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets.items()
    }
    return loaders
