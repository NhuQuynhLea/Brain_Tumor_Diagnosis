# LS-Net (Kyrkou, CVPR 2026): 24-ch stride-2 stem -> MPAC stages with doubling
# channels (downsample at stage start) -> 1x1 conv to num_classes + BN + GAP.

import torch
import torch.nn as nn

from src.models.mpac_block import MPACBlock, make_stage


class LSNet(nn.Module):
    def __init__(self, num_classes=4, widths=(48, 96, 192, 384),
                 blocks=(2, 4, 4, 2), stem_width=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.LeakyReLU(inplace=True),
        )
        in_ch = stem_width
        stages = []
        for w, n in zip(widths, blocks):
            stage, in_ch = make_stage(in_ch, MPACBlock, w, n, stride=2)
            stages.append(stage)
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, num_classes, 1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def build_lsnet(num_classes=4, **_):
    return LSNet(num_classes=num_classes)
