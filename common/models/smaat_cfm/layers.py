"""Building blocks for the SmaAt-CFM backbone: depthwise-separable convolutions,
CBAM attention, and AdaGN-FiLM time conditioning -- ported from the official
SmaAt-UNet architecture (Trebing et al. 2020, github.com/HansBambel/SmaAt-UNet),
with the minimal modifications required to act as a Conditional Flow Matching
velocity-field backbone: BatchNorm -> GroupNorm (batch members sit at different
flow times, so batch statistics would mix incompatible examples), and a new
AdaGN-style time-conditioning module that the original (unconditional,
deterministic) architecture has no equivalent of.
"""

import torch
from torch import nn


def _group_norm(channels, max_groups=32, affine=True):
    """GroupNorm with as many groups as evenly divide `channels` (up to `max_groups`).

    The original SmaAt-UNet uses BatchNorm, which is fine for its deterministic,
    unconditional regression setting. Here the backbone is conditioned on a flow
    time `t` sampled independently per example, so a single training batch mixes
    examples at very different points along the noise-to-data path; BatchNorm's
    shared batch statistics would blend those together. CuboidTransformerUNet
    (the existing FlowCast backbone) avoids this with GroupNorm for the same
    reason -- matching that choice here instead of carrying over BatchNorm
    unexamined from the original (non-conditional) SmaAt-UNet. This mirrors
    Dhariwal & Nichol (2021) and Ho et al. (2020), who likewise avoid BatchNorm
    in diffusion U-Nets for the same batch-heterogeneity reason.
    """
    num_groups = max_groups if channels % max_groups == 0 else channels
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels, affine=affine)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv (per-channel spatial filter) followed by a pointwise (1x1) conv.

    `kernels_per_layer` is the depthwise depth multiplier (the original SmaAt-UNet's
    own term): the depthwise stage produces `in_channels * kernels_per_layer`
    channels before the pointwise conv combines them down to `out_channels`. The
    published SmaAt-UNet precipitation-nowcasting model uses kernels_per_layer=2,
    not the library default of 1 -- matched here for the same reason. Bias terms
    on both convs match the official repo (it does not pass bias=False here).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, kernels_per_layer=2):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


class DoubleConvDS(nn.Module):
    """Two stacked (depthwise-separable conv -> norm -> ReLU) layers, matching the
    official SmaAt-UNet's DoubleConvDS exactly except BatchNorm2d -> GroupNorm."""

    def __init__(self, in_channels, out_channels, kernels_per_layer=2):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernels_per_layer=kernels_per_layer),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernels_per_layer=kernels_per_layer),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    """Matches the official SmaAt-UNet's ChannelAttention (avg-pool and max-pool
    branches through a shared MLP, summed, then sigmoid). The MLP is expressed as
    1x1 convolutions instead of Flatten+Linear -- a mathematically identical
    reformulation for a (B, C, 1, 1) input, not an architectural change."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Matches the official SmaAt-UNet's SpatialAttention, including the
    normalization layer after the conv (BatchNorm2d(1) there; GroupNorm(1, 1)
    here, for the same per-example-t reason as DoubleConvDS above)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.norm(self.conv(pooled)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial
    attention, applied sequentially -- matches the official SmaAt-UNet exactly."""

    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


class AdaGNFiLM(nn.Module):
    """Adaptive Group Normalization (Dhariwal & Nichol, 2021, "Diffusion Models
    Beat GANs on Image Synthesis"): a parameter-free GroupNorm followed by a
    FiLM-style scale-shift computed from the time embedding,
    `GroupNorm(x) * (1 + scale) + shift`. The original SmaAt-UNet has no
    timestep-conditioning mechanism at all (it is a one-shot deterministic
    regressor); this is the literature-preferred way to add one to a conv U-Net,
    outperforming both plain FiLM-on-raw-features and additive conditioning in
    Dhariwal & Nichol's own ablation (FID 13.06 for AdaGN vs 15.08 for additive).
    The projection uses ordinary (non-zero) initialization, matching the ADM
    codebase: ADM's "zero module" stability trick zero-initializes the *final*
    conv of each residual block (so the block's residual contribution starts
    at zero), not the conditioning projection itself -- zeroing this
    projection instead would make the model's output identical for every t at
    initialization, defeating the purpose of conditioning on t at all.
    """

    def __init__(self, time_embed_dim, channels):
        super().__init__()
        self.norm = _group_norm(channels, affine=False)
        self.proj = nn.Linear(time_embed_dim, 2 * channels)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, t_emb):
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return self.norm(x) * (1 + scale) + shift


class Down(nn.Module):
    """Maxpool downsample followed by a double depthwise-separable conv block."""

    def __init__(self, in_channels, out_channels, kernels_per_layer=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample, concatenate with the encoder skip, then a double conv
    block -- matches the official SmaAt-UNet's bilinear=True path exactly."""

    def __init__(self, low_res_channels, skip_channels, out_channels, kernels_per_layer=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvDS(low_res_channels + skip_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
