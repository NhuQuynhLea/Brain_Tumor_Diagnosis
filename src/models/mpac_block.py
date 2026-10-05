# Multi-Path Atrous Convolution block (Kyrkou, CVPR 2026) + shared stage builder.

import torch
import torch.nn as nn


class MPACBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation_rates=(1, 2), kernel_size=3):
        super().__init__()
        self.paths = nn.ModuleList(
            nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride,
                      padding=d * (kernel_size - 1) // 2, dilation=d,
                      groups=inplanes, bias=False)
            for d in dilation_rates
        )
        self.mix = nn.Sequential(
            nn.BatchNorm2d(inplanes * len(dilation_rates)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inplanes * len(dilation_rates), planes * self.expansion, 1, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
        )
        self.downsample = downsample
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x
        out = self.mix(torch.cat([p(x) for p in self.paths], dim=1))
        return self.act(out + residual)


def make_stage(inplanes, block, planes, blocks, stride=1):
    out = planes * block.expansion
    downsample = None
    if stride != 1 or inplanes != out:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, out, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out),
        )
    layers = [block(inplanes, planes, stride, downsample)]
    layers += [block(out, planes) for _ in range(1, blocks)]
    return nn.Sequential(*layers), out
