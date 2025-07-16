from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class InpaintNet(nn.Module):
    """A very small UNet-like network to showcase training pipeline.

    This is **not** the original LaMa architecture – it is a minimal, light-weight
    model that can run quickly in CI for demonstration purposes.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 32):
        """Args:
            in_channels: Number of input channels (RGB + mask = 4 by default)
            out_channels: Number of output channels (RGB = 3)
            base_channels: Number of filters in the first conv layer.
        """
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.conv5 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, image: torch.Tensor, mask: torch.Tensor):  # noqa: D401
        """Forward expects image ∈ [0, 1] and *binary* mask ∈ {0, 1}."""
        x = torch.cat([image, mask], dim=1)

        # Encoder
        e1 = F.relu(self.conv1(x))
        e2 = F.relu(self.conv2(F.max_pool2d(e1, 2)))
        e3 = F.relu(self.conv3(F.max_pool2d(e2, 2)))

        # Decoder
        d1 = F.relu(self.up1(e3))
        d1 = torch.cat([d1, e2], dim=1)
        d1 = F.relu(self.conv4(d1))

        d2 = F.relu(self.up2(d1))
        d2 = torch.cat([d2, e1], dim=1)
        d2 = F.relu(self.conv5(d2))

        out = torch.sigmoid(self.out_conv(d2))
        return out