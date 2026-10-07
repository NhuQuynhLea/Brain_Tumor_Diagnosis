import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.split import describe_splits, grouped_stratified_split, write_manifests
from src.utils.config import load_config
from src.utils.tracking import get_logger

logger = get_logger("prepare_brats", Path("results/logs/prepare_brats.log"))

MODALITIES = ("flair", "t1", "t1ce", "t1n", "t2")


def load_grades(src: Path) -> dict:
    hits = list(src.rglob("*name_mapping*.csv"))
    if not hits:
        raise SystemExit(f"name_mapping.csv not found under {src}")
    df = pd.read_csv(hits[0])
    id_col = next((c for c in df.columns if "subject_id" in c.lower()), df.columns[0])
    grade_col = next(c for c in df.columns if c.strip().lower() == "grade")
    return {str(r[id_col]): str(r[grade_col]).strip().lower() for _, r in df.iterrows()}


def norm_volume(vol: np.ndarray) -> np.ndarray:
    b = vol[vol > 0]
    z = np.clip((vol - b.mean()) / (b.std() + 1e-8), -3, 3)
    return ((z + 3) / 6 * 255).astype(np.uint8)


def export_subject(subj_dir: Path, label: str, modalities: list, out_dir: Path,
                   max_slices: int, min_px: int) -> list[dict]:
    paths = {m: next(subj_dir.glob(f"*_{m}.nii*"), None) for m in [*modalities, "seg"]}
    if any(v is None for v in paths.values()):
        logger.warning("%s: missing modality/seg file, skipped", subj_dir.name)
        return []
    vols = {m: norm_volume(nib.load(str(p)).get_fdata(dtype=np.float32))
            for m, p in paths.items() if m != "seg"}
    seg = nib.load(str(paths["seg"])).get_fdata(dtype=np.float32)
    areas = (seg > 0).sum(axis=(0, 1))
    keep = [i for i in np.argsort(areas)[::-1] if areas[i] >= min_px][:max_slices]
    if not keep:
        logger.warning("%s: no slice with >=%d tumor px", subj_dir.name, min_px)
        return []
    rows = []
    for i in keep:
        img = np.stack([np.rot90(vols[m][:, :, i]) for m in modalities], axis=-1)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        fp = out_dir / f"s{i:03d}.png"
        cv2.imwrite(str(fp), img[..., ::-1])
        rows.append({"filepath": str(fp), "label": label, "subject": subj_dir.name})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="BraTS NIfTI -> modality-stacked slice PNGs + subject-level manifests")
    p.add_argument("--src", required=True,
                   help="BraTS root containing BraTS20_* subject dirs and name_mapping.csv")
    p.add_argument("--config", default="configs/config_brats.yaml")
    p.add_argument("--name", default="brats2020")
    p.add_argument("--img-root", default=None, help="default: data/external_raw/<name>")
    p.add_argument("--out-root", default="data/external")
    p.add_argument("--modalities", nargs="+", default=["flair", "t1ce", "t2"],
                   choices=list(MODALITIES))
    p.add_argument("--max-slices", type=int, default=25)
    p.add_argument("--min-tumor-px", type=int, default=200)
    args = p.parse_args()

    cfg = load_config(args.config)
    classes = list(cfg.data.classes)
    src = Path(args.src)
    grades = load_grades(src)
    img_root = Path(args.img_root or f"data/external_raw/{args.name}")

    rows, skipped = [], []
    subj_dirs = [d for d in src.rglob("*") if d.is_dir() and d.name in grades]
    logger.info("Found %d subject dirs under %s", len(subj_dirs), src)
    for subj_dir in tqdm(subj_dirs, desc="Exporting slices"):
        label = grades[subj_dir.name]
        if label not in classes:
            skipped.append(subj_dir.name)
            continue
        out_dir = img_root / label / subj_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        rows += export_subject(subj_dir, label, args.modalities, out_dir,
                               args.max_slices, args.min_tumor_px)
    if skipped:
        logger.info("Skipped %d subjects with grade outside %s", len(skipped), classes)
    if not rows:
        raise SystemExit("No slices exported - check --src layout")

    df = pd.DataFrame(rows)
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(classes)})
    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all.csv", index=False)
    logger.info("Wrote %s (%d slices, %d subjects)",
                out_dir / "all.csv", len(df), df["subject"].nunique())

    splits = grouped_stratified_split(df, cfg)
    write_manifests(splits, classes, out_dir)
    logger.info("Slice counts:\n%s", describe_splits(splits).to_string())
    subj_counts = {k: s["subject"].nunique() for k, s in splits.items()}
    logger.info("Subject counts: %s", subj_counts)


if __name__ == "__main__":
    main()
