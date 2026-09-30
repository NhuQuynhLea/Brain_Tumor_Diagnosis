"""Stratified train/val/test splitting. Produces manifest CSVs in data/processed/."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    """Split a cleaned DataFrame [filepath, label] into stratified train/val/test."""
    seed = cfg.project.seed
    ratios = cfg.data.split
    test_val = ratios.val + ratios.test
    val_fraction_of_remainder = ratios.val / test_val

    train_df, rem_df = train_test_split(
        df, test_size=test_val, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        rem_df, test_size=1 - val_fraction_of_remainder,
        stratify=rem_df["label"], random_state=seed
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def write_manifests(splits: dict[str, pd.DataFrame], classes: list[str], out_dir: Path) -> None:
    """Add label_idx column and write one CSV per split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for name, df in splits.items():
        out = df.copy()
        out["label_idx"] = out["label"].map(class_to_idx)
        out.to_csv(out_dir / f"{name}.csv", index=False)


def describe_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-split class counts for reporting."""
    return pd.DataFrame(
        {name: df["label"].value_counts().sort_index() for name, df in splits.items()}
    ).fillna(0).astype(int)
