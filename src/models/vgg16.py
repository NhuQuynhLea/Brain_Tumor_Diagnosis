"""VGG16 transfer learning: frozen ImageNet features + new 4-class classifier head."""

import torch.nn as nn
from torchvision import models


def build_vgg16(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.5,
    **_: object,
) -> nn.Module:
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return model
