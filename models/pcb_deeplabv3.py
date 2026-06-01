"""
models/pcb_deeplabv3.py
========================
PCB-DeepLabV3: the complete model from the paper.

Architecture summary
--------------------
Encoder
  MobileNetV2 backbone (lightweight, pretrained on ImageNet)
    ├─ bottleneck-group 3 → 3 low-level feature maps [B, 32, H/8, W/8]
    │    └─→ AMTPNet fuses them         → [B, 288, H/8, W/8]
    └─ bottleneck-group 6 → high-level  → [B, 160, H/32, W/32]
         └─→ ASPP                       → [B, 256, H/32, W/32]

Decoder
  1×1 conv: AMTPNet output 288 → 48
  Upsample ASPP output to low-level spatial size
  Concat (256 + 48 = 304) → two 3×3 conv blocks → 1×1 classifier
  Final bilinear upsample to original input resolution

MobileNetV2 feature-index reference (torchvision, 512×512 input)
  .features[0..3]   stem + groups 1-2     → 128×128
  .features[4]      group-3 layer 0       →  64× 64 ← f1
  .features[5]      group-3 layer 1       →  64× 64 ← f2
  .features[6]      group-3 layer 2       →  64× 64 ← f3
  .features[7..16]  groups 4-6            →  16× 16, 160ch ← ASPP input
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.amtpnet import AMTPNet
from models.aspp    import ASPP


class _CBR(nn.Sequential):
    """Conv2d + BatchNorm + ReLU."""
    def __init__(self, cin, cout, k=3, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class PCBDeepLabV3(nn.Module):
    """
    Parameters
    ----------
    num_classes : output classes (2: background + void)
    pretrained  : initialise backbone with ImageNet weights
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        # ── MobileNetV2 backbone ──────────────────────────────────────────────
        weights   = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        feats     = mobilenet_v2(weights=weights).features

        # stem + groups 1-2  (indices 0-3)
        self.early = nn.Sequential(*list(feats.children())[:4])

        # bottleneck-group 3  (three separate layers at indices 4, 5, 6)
        self.bn3_0 = feats[4]
        self.bn3_1 = feats[5]
        self.bn3_2 = feats[6]

        # groups 4-6  (indices 7-16) → 16×16×160 for 512-input
        self.deep  = nn.Sequential(*list(feats.children())[7:17])

        # ── AMTPNet on the three low-level feature maps ───────────────────────
        self.amtpnet = AMTPNet(
            in_channels  = config.LOW_LEVEL_CHANNELS,    # 32
            out_channels = config.AMTPNET_OUT_CHANNELS,  # 288
        )

        # ── ASPP on high-level features ───────────────────────────────────────
        self.aspp = ASPP(
            in_channels  = config.HIGH_LEVEL_CHANNELS,   # 160
            out_channels = config.ASPP_OUT_CHANNELS,     # 256
            atrous_rates = config.ASPP_ATROUS_RATES,     # [6,12,18]
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # 1×1 conv: reduce 288 → 48  (paper's "too many channels will mask importance")
        self.low_proj = _CBR(config.AMTPNET_OUT_CHANNELS,
                             config.LOW_LEVEL_REDUCED, k=1, p=0)

        # Two 3×3 conv blocks on concatenated (256 + 48 = 304) feature maps
        dec_in = config.ASPP_OUT_CHANNELS + config.LOW_LEVEL_REDUCED  # 304
        self.decoder = nn.Sequential(
            _CBR(dec_in, 256, k=3, p=1),
            nn.Dropout2d(0.1),
            _CBR(256, 256, k=3, p=1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

        self._init_new_layers()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_new_layers(self):
        for module in [self.amtpnet, self.aspp, self.low_proj, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : [B, 3, H, W]
        returns : [B, num_classes, H, W]  (same spatial size as input)
        """
        size = x.shape[-2:]    # original (H, W) for final upsample

        # ── Encoder ───────────────────────────────────────────────────────────
        x  = self.early(x)     # [B, 24, H/4, W/4]

        f1 = self.bn3_0(x)     # [B, 32, H/8, W/8]
        f2 = self.bn3_1(f1)    # [B, 32, H/8, W/8]
        f3 = self.bn3_2(f2)    # [B, 32, H/8, W/8]

        low  = self.amtpnet(f1, f2, f3)   # [B, 288, H/8, W/8]

        high = self.deep(f3)              # [B, 160, H/32, W/32]
        high = self.aspp(high)            # [B, 256, H/32, W/32]

        # ── Decoder ───────────────────────────────────────────────────────────
        low  = self.low_proj(low)          # [B, 48, H/8, W/8]

        high = F.interpolate(high, size=low.shape[-2:],
                             mode="bilinear", align_corners=False)  # [B,256,H/8,W/8]

        x = torch.cat([high, low], dim=1)  # [B, 304, H/8, W/8]
        x = self.decoder(x)                # [B, num_classes, H/8, W/8]

        return F.interpolate(x, size=size,
                             mode="bilinear", align_corners=False)  # [B, C, H, W]

    # ── Utilities ─────────────────────────────────────────────────────────────
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """Freeze MobileNetV2 weights — useful when dataset is very small."""
        for part in [self.early, self.bn3_0, self.bn3_1, self.bn3_2, self.deep]:
            for p in part.parameters():
                p.requires_grad = False
        print("Backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True
        print("All parameters unfrozen.")


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = PCBDeepLabV3(num_classes=2, pretrained=False)
    dummy = torch.zeros(1, 3, 512, 512)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")          # should be [1, 2, 512, 512]
    print(f"Params: {model.count_params():,}")
