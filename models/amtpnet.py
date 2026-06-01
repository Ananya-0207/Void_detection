"""
models/amtpnet.py
==================
AMTPNet: Attentional Multi-scale Two-space Pyramid Pooling Network.

Paper §"Attentional multi-scale two-space pyramid pooling network design":

  Four parts:
  ① Horizontal pyramid  — one branch per input feature map;
                          each branch applies 1×1, 3×3, 5×5 convs in parallel
                          then applies channel attention → [B, 3C, H, W]
  ② Longitudinal concat — concatenate all three branches → [B, 9C, H, W]
  ③ Channel attention   — SENet on the full 9C feature map
  ④ Channel adjustment  — 1×1 conv → out_channels (288 by default)
"""

import torch
import torch.nn as nn
from models.channel_attention import ChannelAttention


class _CBR(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU."""
    def __init__(self, cin, cout, k=1, p=0):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class _HorizontalBranch(nn.Module):
    """
    One horizontal pyramid branch.
    Applies 1×1, 3×3, 5×5 convolutions on a single feature map,
    concatenates them, then applies channel attention.
    Output: [B, 3C, H, W]
    """
    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        self.conv1 = _CBR(c, c, k=1, p=0)
        self.conv3 = _CBR(c, c, k=3, p=1)
        self.conv5 = _CBR(c, c, k=5, p=2)
        self.attn  = ChannelAttention(c * 3, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.conv1(x),
                         self.conv3(x),
                         self.conv5(x)], dim=1)   # [B, 3C, H, W]
        return self.attn(out)


class AMTPNet(nn.Module):
    """
    Full AMTPNet module.

    Parameters
    ----------
    in_channels  : channels of each of the 3 input feature maps (32 for MBv2 group-3)
    out_channels : final output channels (paper: 288 = 9×32)
    reduction    : squeeze ratio for ChannelAttention
    """

    def __init__(self, in_channels: int = 32,
                       out_channels: int = 288,
                       reduction:    int = 4):
        super().__init__()
        # ① Three horizontal pyramid branches (one per input feature map)
        self.b1 = _HorizontalBranch(in_channels, reduction)
        self.b2 = _HorizontalBranch(in_channels, reduction)
        self.b3 = _HorizontalBranch(in_channels, reduction)

        # ③ Channel attention on concatenated 9C feature map
        nine_c = in_channels * 9
        self.longitudinal_attn = ChannelAttention(nine_c, reduction=reduction)

        # ④ Channel adjustment to out_channels
        self.adjust = _CBR(nine_c, out_channels, k=1, p=0)

    def forward(self, f1: torch.Tensor,
                       f2: torch.Tensor,
                       f3: torch.Tensor) -> torch.Tensor:
        """
        f1, f2, f3 : three feature maps from MBv2 bottleneck-group-3,
                     each [B, 32, H/8, W/8]
        Returns    : [B, out_channels, H/8, W/8]
        """
        p1 = self.b1(f1)                             # [B, 3C, H, W]
        p2 = self.b2(f2)                             # [B, 3C, H, W]
        p3 = self.b3(f3)                             # [B, 3C, H, W]

        fused = torch.cat([p1, p2, p3], dim=1)       # [B, 9C, H, W]
        fused = self.longitudinal_attn(fused)         # ③
        return self.adjust(fused)                     # ④ → [B, out_channels, H, W]
