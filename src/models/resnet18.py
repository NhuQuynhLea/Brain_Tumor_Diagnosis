

import torch.nn as nn
from torchvision import models


def build_resnet18(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.5,
    **_: object,
) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
