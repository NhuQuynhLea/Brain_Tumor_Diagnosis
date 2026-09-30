"""Phase 4 smoke test: build every registered model, run forward + backward on a real batch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import build_dataloaders
from src.models.model_factory import available_models, build_model, count_parameters
from src.training.losses import build_criterion, build_optimizer, compute_class_weights
from src.utils.config import load_config, seed_everything
from src.utils.tracking import get_logger

logger = get_logger("model_smoke", Path("results/logs/model_smoke.log"))


def main() -> None:
    cfg = load_config()
    seed_everything(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | models: %s", device, available_models())

    images, labels = next(iter(build_dataloaders(cfg)["train"]))
    images, labels = images[:4].to(device), labels[:4].to(device)

    weights = compute_class_weights(Path(cfg.paths.processed_data) / "train.csv", len(cfg.data.classes))
    logger.info("Class weights from train.csv: %s", weights.numpy().round(3).tolist())

    for name in available_models():
        model = build_model(cfg, name=name).to(device)
        counts = count_parameters(model)

        out = model(images)
        assert out.shape == (4, len(cfg.data.classes)), f"{name}: bad output {out.shape}"

        criterion = build_criterion(cfg)
        opt = build_optimizer(cfg, model)
        loss = criterion(out, labels)
        opt.zero_grad(); loss.backward(); opt.step()

        logger.info(
            "%-15s out=%s loss=%.4f | params: total=%d trainable=%d frozen=%d",
            name, tuple(out.shape), loss.item(),
            counts["total"], counts["trainable"], counts["frozen"],
        )
        del model

    logger.info("All models PASSED.")


if __name__ == "__main__":
    main()
