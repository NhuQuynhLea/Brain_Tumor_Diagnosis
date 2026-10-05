"""Custom CNN baseline built from scratch (4 conv blocks + classifier head)."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """Baseline CNN: 4 ConvBlocks -> GAP -> FC -> dropout -> FC.

    Input: (B, 3, 224, 224). Output: (B, num_classes) logits.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.5,
                 widths: tuple = (32, 64, 128, 256)):
        super().__init__()
        blocks = [ConvBlock(3, widths[0])]
        blocks += [ConvBlock(a, b) for a, b in zip(widths, widths[1:])]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1], widths[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_custom_cnn(num_classes: int = 4, dropout: float = 0.5,
                     widths: tuple = (32, 64, 128, 256), **_: object) -> CustomCNN:
    return CustomCNN(num_classes=num_classes, dropout=dropout, widths=widths)


def build_custom_cnn_tiny(num_classes: int = 4, dropout: float = 0.5, **_: object) -> CustomCNN:
    return CustomCNN(num_classes=num_classes, dropout=dropout, widths=(16, 32, 64, 128))
