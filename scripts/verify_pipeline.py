"""Task 3.8 - End-to-end data pipeline verification.

Checks: manifests exist, datasets load, batch shapes/labels correct,
augmentation produces varied outputs, dataloaders iterate.
Saves results/figures/augmented_samples.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.dataset import build_dataloaders
from src.data.preprocessing import denormalize
from src.utils.config import load_config, seed_everything
from src.utils.tracking import get_logger

logger = get_logger("verify", Path("results/logs/verify.log"))


def main() -> None:
    cfg = load_config()
    seed_everything(cfg.project.seed)
    loaders = build_dataloaders(cfg)

    for split, loader in loaders.items():
        logger.info("%s: %d batches x %d", split, len(loader), cfg.training.batch_size)

    images, labels = next(iter(loaders["train"]))
    assert images.ndim == 4 and images.shape[1] == 3, f"Bad batch shape {images.shape}"
    assert images.shape[2] == cfg.data.image_size and images.shape[3] == cfg.data.image_size
    assert images.dtype == torch.float32, f"Bad dtype {images.dtype}"
    assert labels.min() >= 0 and labels.max() < len(cfg.data.classes)
    logger.info("Batch shape %s, dtype %s, label range [%d, %d] - OK",
                tuple(images.shape), images.dtype, labels.min().item(), labels.max().item())

    # Augmentation variability: same dataset item fetched twice should differ
    ds = loaders["train"].dataset
    a, _ = ds[0]
    b, _ = ds[0]
    diff = (a - b).abs().mean().item()
    logger.info("Augmentation variability (mean abs diff of 2 draws): %.4f", diff)
    assert diff > 1e-4, "Augmentation appears inactive!"

    # Visualize one augmented batch
    n = 8
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 6))
    for ax, img, lab in zip(axes.flat, images[:n], labels[:n]):
        ax.imshow(denormalize(img, cfg.preprocessing.normalize.mean, cfg.preprocessing.normalize.std))
        ax.set_title(cfg.data.classes[lab.item()], fontsize=9)
        ax.axis("off")
    fig.suptitle("Augmented Training Batch")
    fig.tight_layout()
    out = Path(cfg.paths.figures) / "augmented_samples.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    logger.info("Data pipeline verification PASSED.")


if __name__ == "__main__":
    main()
