"""Task 3.3 - Data cleaning & quality check.

Scans data/raw, validates every image decodes, drops exact duplicates,
and writes data/processed/clean.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import drop_duplicates, quality_check, scan_dataset
from src.utils.config import load_config
from src.utils.tracking import get_logger

logger = get_logger("cleaning", Path("results/logs/cleaning.log"))


def main() -> None:
    cfg = load_config()
    out_dir: Path = cfg.paths.processed_data
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(cfg.paths.raw_data, cfg.data.classes)
    logger.info("Scanned %d images.", len(df))
    logger.info("Per-class counts:\n%s", df["label"].value_counts().to_string())

    df, corrupted = quality_check(df)
    logger.info("Corrupted/unreadable images removed: %d", len(corrupted))

    df, dupes = drop_duplicates(df)
    logger.info("Exact-duplicate images removed: %d", len(dupes))
    if len(dupes):
        dupes.to_csv(out_dir / "duplicates_removed.csv", index=False)

    df.to_csv(out_dir / "clean.csv", index=False)
    logger.info("Clean dataset: %d images -> %s", len(df), out_dir / "clean.csv")


if __name__ == "__main__":
    main()
