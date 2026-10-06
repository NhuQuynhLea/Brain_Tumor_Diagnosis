# MPAC-ResNet: ResNet-18 stem + stages 1-2 kept (optionally ImageNet-pretrained),
# stages 3-4 replaced with MPAC blocks. CompactResNet powers the tiny variants.

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock

from src.models.mpac_block import MPACBlock, make_stage


class CompactResNet(nn.Module):
    def __init__(self, block, layers, widths, num_classes=4, dropout=0.5, stem_width=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.LeakyReLU(inplace=True),
        )
        in_ch = stem_width
        stages = []
        for i, (w, n) in enumerate(zip(widths, layers)):
            stage, in_ch = make_stage(in_ch, block, w, n, stride=1 if i == 0 else 2)
            stages.append(stage)
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(in_ch, num_classes),
        )

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def build_mpac_resnet(num_classes=4, pretrained=True, freeze_backbone=True,
                      dropout=0.5, **_):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.layer3, _ = make_stage(128, MPACBlock, 256, 2, stride=2)
    model.layer4, _ = make_stage(256, MPACBlock, 512, 2, stride=2)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    if freeze_backbone:
        for name in ("conv1", "bn1", "layer1", "layer2"):
            for p in getattr(model, name).parameters():
                p.requires_grad = False
    return model


def build_mpac_resnet_tiny(num_classes=4, dropout=0.5,
                           widths=(48, 96, 192, 384), layers=(2, 2, 2, 2), **_):
    return CompactResNet(MPACBlock, layers, widths, num_classes, dropout)


def build_resnet_tiny(num_classes=4, dropout=0.5,
                      widths=(24, 48, 96, 192), layers=(4, 3, 2, 1), **_):
    return CompactResNet(BasicBlock, layers, widths, num_classes, dropout)
