"""Task 7.2 - Build eval/train manifests for an external dataset (cross-dataset study).

    python scripts/prepare_external.py --src D:/datasets/figshare --name figshare
    python scripts/prepare_external.py --kaggle <handle> --name figshare
    python scripts/prepare_external.py --src ... --name figshare --split --label-map "1=meningioma,2=glioma,3=pituitary"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import drop_duplicates, quality_check, scan_dataset
from src.data.split import describe_splits, stratified_split, write_manifests
from src.utils.config import load_config
from src.utils.tracking import get_logger

logger = get_logger("prepare_external", Path("results/logs/prepare_external.log"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=None, help="Dataset root containing <class>/** images")
    p.add_argument("--kaggle", default=None, help="kagglehub dataset handle (downloaded to cache)")
    p.add_argument("--name", required=True, help="Dataset name -> data/external/<name>/")
    p.add_argument("--out-root", default="data/external")
    p.add_argument("--split", action="store_true",
                   help="Also write stratified train/val/test.csv (for task 7.5 fine-tuning)")
    p.add_argument("--label-map", default=None,
                   help="Comma list of raw_folder=class, e.g. '1=meningioma,2=glioma'")
    args = p.parse_args()

    cfg = load_config()
    classes = (dict(kv.split("=") for kv in args.label_map.split(","))
               if args.label_map else list(cfg.data.classes))

    src = Path(args.src) if args.src else None
    if args.kaggle:
        import kagglehub
        src = Path(kagglehub.dataset_download(args.kaggle))
        logger.info("Downloaded %s -> %s", args.kaggle, src)
    if src is None or not src.exists():
        raise SystemExit("Provide --src <dir> or --kaggle <handle>")

    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(src, classes)
    if df.empty:
        raise SystemExit(
            f"No images under {src} matched classes {list(classes)}. "
            "Check folder names or pass --label-map.")
    logger.info("Scanned %d images:\n%s", len(df), df["label"].value_counts().to_string())

    df, corrupted = quality_check(df)
    df, dupes = drop_duplicates(df)
    logger.info("Removed: %d corrupted, %d duplicates", len(corrupted), len(dupes))

    df = df.assign(label_idx=df["label"].map({c: i for i, c in enumerate(cfg.data.classes)}))
    bad = df[df["label_idx"].isna()]["label"].unique()
    if len(bad):
        raise SystemExit(f"Labels not in config classes: {list(bad)} - fix --label-map")

    df.to_csv(out_dir / "all.csv", index=False)
    logger.info("Wrote %s (%d images)", out_dir / "all.csv", len(df))

    if args.split:
        rem = cfg.data.split.val + cfg.data.split.test
        min_needed = int(-(-2 // rem))  # ceil(2/rem): each class needs >=2 samples in the val+test remainder
        if df["label"].value_counts().min() < min_needed:
            logger.warning("Need >=%d images/class for stratified split - only all.csv written", min_needed)
        else:
            splits = stratified_split(df, cfg)
            write_manifests(splits, cfg.data.classes, out_dir)
            logger.info("Splits:\n%s", describe_splits(splits).to_string())


if __name__ == "__main__":
    main()
