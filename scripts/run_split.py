"""Task 3.5 - Stratified train/val/test split. Writes train.csv / val.csv / test.csv manifests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.split import describe_splits, stratified_split, write_manifests
from src.utils.config import load_config
from src.utils.tracking import get_logger

logger = get_logger("split", Path("results/logs/split.log"))


def main() -> None:
    cfg = load_config()
    processed: Path = cfg.paths.processed_data

    df = pd.read_csv(processed / "clean.csv")
    splits = stratified_split(df, cfg)
    write_manifests(splits, cfg.data.classes, processed)

    summary = describe_splits(splits)
    summary.loc["TOTAL"] = summary.sum()
    summary.to_csv(cfg.paths.metrics / "split_distribution.csv")
    logger.info("Split distribution:\n%s", summary.to_string())
    for name, sdf in splits.items():
        logger.info("%s: %d images", name, len(sdf))


if __name__ == "__main__":
    main()
