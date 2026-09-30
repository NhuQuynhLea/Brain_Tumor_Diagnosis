"""Download the Kaggle Brain Tumor MRI dataset into data/raw/.

Requires Kaggle credentials at ~/.kaggle/kaggle.json.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kagglehub

from src.utils.config import load_config
from src.utils.tracking import get_logger

logger = get_logger("download")


def main() -> None:
    cfg = load_config()
    raw_dir: Path = cfg.paths.raw_data

    existing = [p for p in raw_dir.rglob("*") if p.is_file()] if raw_dir.exists() else []
    if existing:
        logger.info("data/raw already contains %d files - skipping download.", len(existing))
        return

    logger.info("Downloading %s via kagglehub...", cfg.data.dataset_handle)
    downloaded = Path(kagglehub.dataset_download(cfg.data.dataset_handle))
    logger.info("Downloaded to cache: %s", downloaded)

    raw_dir.mkdir(parents=True, exist_ok=True)
    for item in downloaded.iterdir():
        dest = raw_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    count = sum(1 for p in raw_dir.rglob("*") if p.is_file())
    logger.info("Copied %d files into %s", count, raw_dir)


if __name__ == "__main__":
    main()
