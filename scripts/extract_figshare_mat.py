"""Task 7.2 - Extract the Figshare/Cheng .mat dataset to grayscale PNGs.

Some Kaggle mirrors (e.g. denizkavi1/brain-tumor) baked a viridis colormap into
the PNGs - unusable for models trained on grayscale MRI. The .mat version
(ashkhagan/figshare-brain-tumor-dataset) keeps the raw cjdata.image arrays.

    python scripts/extract_figshare_mat.py --kaggle ashkhagan/figshare-brain-tumor-dataset --out data/external_raw/figshare
    python scripts/extract_figshare_mat.py --src <dir_with_mat_files> --out data/external_raw/figshare
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from src.utils.tracking import get_logger

logger = get_logger("figshare_extract", Path("results/logs/figshare_extract.log"))

FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def read_cjdata(path: Path) -> dict | None:
    try:
        return loadmat(path)["cjdata"]
    except NotImplementedError:  # MATLAB v7.3 = HDF5
        import h5py
        with h5py.File(path, "r") as f:
            if "cjdata" not in f:
                return None  # e.g. cvind.mat (cross-validation indices)
            d = f["cjdata"]
            return {"label": np.asarray(d["label"]).item(),
                    "image": np.asarray(d["image"]).T}  # HDF5 stores transposed
    except KeyError:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=None, help="Directory containing .mat files")
    p.add_argument("--kaggle", default=None, help="kagglehub handle for the .mat dataset")
    p.add_argument("--out", default="data/external_raw/figshare")
    args = p.parse_args()

    src = Path(args.src) if args.src else None
    if args.kaggle:
        import kagglehub
        src = Path(kagglehub.dataset_download(args.kaggle))
        logger.info("Downloaded %s -> %s", args.kaggle, src)
    if src is None or not src.exists():
        raise SystemExit("Provide --src <dir> or --kaggle <handle>")

    out = Path(args.out)
    mats = sorted(src.rglob("*.mat"))
    if not mats:
        raise SystemExit(f"No .mat files under {src}")

    counts = {c: 0 for c in FIGSHARE_LABELS.values()}
    for mp in tqdm(mats, desc="Extracting .mat"):
        cj = read_cjdata(mp)
        if cj is None:
            continue
        label = int(np.asarray(cj["label"]).item())
        image = np.asarray(cj["image"])
        if image.dtype != np.uint8 or image.max() > 255:
            image = cv2.normalize(image.astype(np.float32), None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8)
        cls = FIGSHARE_LABELS[label]
        dest = out / cls
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest / f"{mp.stem}.png"), image)
        counts[cls] += 1
    logger.info("Extracted %d images to %s: %s", sum(counts.values()), out, counts)


if __name__ == "__main__":
    main()
