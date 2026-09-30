"""Model registry: instantiate any architecture by name from config."""

import torch.nn as nn

from src.models.custom_cnn import build_custom_cnn
from src.models.efficientnet import build_efficientnet
from src.models.mobilenet import build_mobilenet
from src.models.resnet50 import build_resnet50
from src.models.vgg16 import build_vgg16

MODEL_REGISTRY = {
    "custom_cnn": build_custom_cnn,
    "vgg16": build_vgg16,
    "resnet50": build_resnet50,
    "efficientnetb0": build_efficientnet,
    "mobilenetv2": build_mobilenet,
}


def available_models() -> list[str]:
    return list(MODEL_REGISTRY)


def build_model(cfg, name: str | None = None, **overrides) -> nn.Module:
    """Build a model from config. `name` overrides cfg.model.name."""
    name = (name or cfg.model.name).lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")

    kwargs = {
        "num_classes": cfg.data.num_classes if "num_classes" in cfg.data else len(cfg.data.classes),
        "pretrained": cfg.model.pretrained,
        "freeze_backbone": cfg.model.freeze_backbone,
        "dropout": cfg.model.dropout,
        **overrides,
    }
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
