"""Task 3.2 - Exploratory Data Analysis.

Generates into results/figures:
  - class_distribution.png
  - image_size_distribution.png
  - sample_grid.png (sample images per class)
  - intensity_histogram.png
and results/metrics/eda_summary.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.data.clean import scan_dataset
from src.utils.config import load_config
from src.utils.tracking import get_logger

logger = get_logger("eda", Path("results/logs/eda.log"))


def main() -> None:
    cfg = load_config()
    fig_dir: Path = cfg.paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = scan_dataset(cfg.paths.raw_data, cfg.data.classes)
    logger.info("Total images: %d", len(df))

    # --- 1. Class distribution ---
    counts = df["label"].value_counts().reindex(cfg.data.classes)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("viridis", 4))
    ax.bar_label(bars)
    ax.set_title("Class Distribution"); ax.set_ylabel("Images")
    fig.tight_layout(); fig.savefig(fig_dir / "class_distribution.png", dpi=150); plt.close(fig)

    # --- 2. Image dimensions ---
    sizes = df["filepath"].sample(min(500, len(df)), random_state=cfg.project.seed).apply(
        lambda fp: Image.open(fp).size
    )
    size_df = pd.DataFrame(sizes.tolist(), columns=["width", "height"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(size_df["width"], size_df["height"], alpha=0.4, s=12)
    ax.set_title("Image Size Distribution (500-sample)"); ax.set_xlabel("width"); ax.set_ylabel("height")
    fig.tight_layout(); fig.savefig(fig_dir / "image_size_distribution.png", dpi=150); plt.close(fig)

    # --- 3. Sample grid: 4 per class ---
    fig, axes = plt.subplots(len(cfg.data.classes), 4, figsize=(10, 2.6 * len(cfg.data.classes)))
    for i, cls in enumerate(cfg.data.classes):
        samples = df[df["label"] == cls]["filepath"].head(4)
        for j, fp in enumerate(samples):
            axes[i, j].imshow(Image.open(fp), cmap="gray")
            axes[i, j].axis("off")
            axes[i, j].set_title(cls if j == 0 else "")
    fig.suptitle("Sample MRI Images per Class", y=1.0)
    fig.tight_layout(); fig.savefig(fig_dir / "sample_grid.png", dpi=150); plt.close(fig)

    # --- 4. Pixel intensity histogram (grayscale, 300-sample) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in cfg.data.classes:
        pixels = []
        for fp in df[df["label"] == cls]["filepath"].sample(
            min(75, len(df[df["label"] == cls])), random_state=cfg.project.seed
        ):
            pixels.append(np.asarray(Image.open(fp).convert("L")).ravel())
        ax.hist(np.concatenate(pixels), bins=64, alpha=0.4, label=cls, density=True)
    ax.set_title("Pixel Intensity Distribution"); ax.set_xlabel("Intensity (0-255)"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "intensity_histogram.png", dpi=150); plt.close(fig)

    # --- summary table ---
    summary = pd.DataFrame({
        "class": counts.index, "count": counts.values,
        "pct": (counts.values / counts.sum() * 100).round(2),
    })
    summary.to_csv(cfg.paths.metrics / "eda_summary.csv", index=False)
    logger.info("EDA figures saved to %s\n%s", fig_dir, summary.to_string(index=False))


if __name__ == "__main__":
    main()
