"""
models/aspp.py
==============
Standard DeepLabV3 ASPP module.

Five parallel branches:
  1×1 conv  +  three dilated 3×3 convs  +  global avg pool branch
→ concatenated and projected to out_channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CBR(nn.Sequential):
    def __init__(self, cin, cout, k=1, p=0, d=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class _GlobalPoolBranch(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _CBR(cin, cout)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.conv(self.pool(x))
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Parameters
    ----------
    in_channels   : backbone output channels (160 for MBv2 group-6)
    out_channels  : per-branch and final projection channels (256)
    atrous_rates  : dilation rates for the three dilated branches
    dropout       : dropout probability in the projection layer
    """

    def __init__(self, in_channels=160, out_channels=256,
                 atrous_rates=None, dropout=0.5):
        super().__init__()
        if atrous_rates is None:
            atrous_rates = [6, 12, 18]

        self.b1 = _CBR(in_channels, out_channels)                       # 1×1
        self.b2 = _CBR(in_channels, out_channels, k=3, p=atrous_rates[0], d=atrous_rates[0])
        self.b3 = _CBR(in_channels, out_channels, k=3, p=atrous_rates[1], d=atrous_rates[1])
        self.b4 = _CBR(in_channels, out_channels, k=3, p=atrous_rates[2], d=atrous_rates[2])
        self.b5 = _GlobalPoolBranch(in_channels, out_channels)

        self.project = nn.Sequential(
            _CBR(out_channels * 5, out_channels),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.b1(x), self.b2(x),
                    self.b3(x), self.b4(x), self.b5(x)]
        return self.project(torch.cat(branches, dim=1))
