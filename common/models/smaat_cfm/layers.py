"""Building blocks for the SmaAt-CFM backbone: depthwise-separable convolutions,
CBAM attention, and FiLM time conditioning -- following the original SmaAt-UNet
architecture (Trebing et al. 2020), adapted for FiLM-based flow-time conditioning.
"""

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv (per-channel spatial filter) followed by a pointwise (1x1) conv."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class DoubleDSConv(nn.Module):
    """Two stacked depthwise-separable conv layers, as used at every U-Net depth level.

    A residual shortcut wraps both layers: a stack of these blocks (no skip connections
    of its own otherwise) was found to collapse to dead units / vanishing gradients on
    the contract test's synthetic convergence check, so the residual path is required
    for trainability, not just a speed optimization.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return h + self.skip(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(pooled))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial attention."""

    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


class FiLM(nn.Module):
    """Scale-shift (FiLM) conditioning on a time embedding, applied per feature channel."""

    def __init__(self, time_embed_dim, channels):
        super().__init__()
        self.proj = nn.Linear(time_embed_dim, 2 * channels)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, t_emb):
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


class Down(nn.Module):
    """Maxpool downsample followed by a double depthwise-separable conv block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleDSConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample, concatenate with the encoder skip, then a double conv block."""

    def __init__(self, low_res_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleDSConv(low_res_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
