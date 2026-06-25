import torch.nn as nn
import torch.nn.functional as F


class ReliabilityHead(nn.Module):
    """Lightweight deploy-time reliability predictor.

    It learns to approximate the frozen-SAM SSRF reliability map during training,
    then replaces SAM-derived reliability at validation and inference time.
    """

    def __init__(self, in_channels):
        super().__init__()
        hidden = max(in_channels // 2, 16)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat, out_size):
        reliability = self.head(feat)
        return F.interpolate(reliability, size=out_size, mode="bilinear", align_corners=False)
