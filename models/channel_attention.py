"""
models/channel_attention.py
============================
SENet-style Squeeze-and-Excitation channel attention.

Paper §"Channel attention mechanism":
  Compress global spatial info → learn per-channel importance weights
  → re-weight (excite) the feature map.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention block.

    Parameters
    ----------
    in_channels : number of input channels
    reduction   : squeeze ratio (default 4; smaller is safer for narrow feature maps)
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, in_channels // reduction)

        self.squeeze = nn.AdaptiveAvgPool2d(1)   # global average pooling
        self.excite  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.squeeze(x).view(b, c)        # [B, C]
        w = self.excite(s).view(b, c, 1, 1)   # [B, C, 1, 1]
        return x * w                           # element-wise scale
