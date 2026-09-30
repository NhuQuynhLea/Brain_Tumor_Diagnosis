"""Data cleaning & quality check: verify images, drop corrupted/duplicate files."""

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan_dataset(raw_dir: Path, classes: list[str]) -> pd.DataFrame:
    """Scan raw_dir/**/<class>/* images into a DataFrame [filepath, label, source_split]."""
    rows = []
    for img_path in sorted(raw_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label = img_path.parent.name.lower()
        if label not in classes:
            continue
        rows.append(
            {
                "filepath": str(img_path),
                "label": label,
                "source_split": img_path.parent.parent.name.lower(),
            }
        )
    return pd.DataFrame(rows)


def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def quality_check(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify each file opens as a valid image; return (clean_df, corrupted_df)."""
    corrupted = []
    keep_mask = []
    for fp in tqdm(df["filepath"], desc="Validating images"):
        try:
            with Image.open(fp) as im:
                im.verify()
            keep_mask.append(True)
        except Exception:
            keep_mask.append(False)
            corrupted.append(fp)
    return df[keep_mask].reset_index(drop=True), pd.DataFrame({"filepath": corrupted})


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove files with identical content (md5). Keep first occurrence."""
    hashes = [md5_of_file(Path(fp)) for fp in tqdm(df["filepath"], desc="Hashing files")]
    df = df.assign(md5=hashes)
    dup_mask = df.duplicated(subset="md5", keep="first")
    dupes = df[dup_mask].copy()
    return df[~dup_mask].drop(columns="md5").reset_index(drop=True), dupes
