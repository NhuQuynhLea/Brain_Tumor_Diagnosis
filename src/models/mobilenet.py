"""MobileNetV2 transfer learning: frozen features + new classifier head."""

import torch.nn as nn
from torchvision import models


def build_mobilenet(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.5,
    **_: object,
) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_mobilenet_tiny(
    num_classes: int = 4,
    dropout: float = 0.5,
    width_mult: float = 0.35,
    **_: object,
) -> nn.Module:
    return models.mobilenet_v2(weights=None, width_mult=width_mult,
                               num_classes=num_classes, dropout=dropout)
