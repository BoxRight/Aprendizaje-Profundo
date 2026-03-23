"""Small 2D CNN for single-channel Doppler spectrograms."""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallRadarCNN(nn.Module):
    """
    ~1–2M parameters default (narrow channels).
    Input: (N, 1, H, W)
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        base: int = 48,
        use_head_bn: bool = False,
    ) -> None:
        super().__init__()
        b = base
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.b1 = _conv_block(b, b)
        self.b2 = _conv_block(b, b * 2)
        self.b3 = _conv_block(b * 2, b * 4)
        self.b4 = _conv_block(b * 4, b * 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat = b * 4
        self.use_head_bn = use_head_bn
        self.head_bn = nn.BatchNorm1d(feat) if use_head_bn else nn.Identity()
        self.fc = nn.Linear(feat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.gap(x).flatten(1)
        x = self.head_bn(x)
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
