"""EfficientNetB0 transfer learning with custom classifier head."""

import torch.nn as nn
from torchvision import models


def build_efficientnet(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.5,
    **_: object,
) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model
