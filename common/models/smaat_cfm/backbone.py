"""SmaAt-CFM backbone: a SmaAt-UNet-style (depthwise-separable conv + CBAM) drop-in
replacement for CuboidTransformerUNet in the Conditional Flow Matching framework.

Implements the same forward(t, x, cond) contract as
`common.models.flowcast.cuboid_transformer_unet.CuboidTransformerUNet`:
  - t:    flow time, shape (B,)
  - x:    (partially noised) future latent state, shape (B, T_out, H, W, C)
  - cond: past latent conditioning sequence, shape (B, T_in, H, W, C)
  - returns: velocity field, shape (B, T_out, H, W, C), unbounded (no output activation)

Past conditioning (`cond`) is fused via channel-stacking (flattening the temporal
dimension into channels) rather than 3D/attention-based temporal mixing, matching the
original SmaAt-UNet's pure-2D, early-fusion philosophy. Flow time `t` is injected via a
sinusoidal embedding + small MLP, applied with FiLM (scale-shift) at every encoder and
decoder depth level.
"""

import torch
from einops import rearrange
from torch import nn

from common.models.flowcast.utils import timestep_embedding
from common.models.smaat_cfm.layers import CBAM, DoubleDSConv, Down, FiLM, Up


class SmaatCFMBackbone(nn.Module):
    def __init__(
        self,
        input_shape,
        target_shape,
        base_channels=32,
        depth=5,
        time_embed_dim=128,
        cbam_reduction=16,
        mean=0.0,
        std=1.0,
    ):
        """
        Parameters
        ----------
        input_shape: tuple
            (T_in, H, W, C) shape of the conditioning (`cond`) latent sequence,
            derived at runtime from the actual data -- not hardcoded.
        target_shape: tuple
            (T_out, H, W, C) shape of the (noised) future state (`x`) / output.
        base_channels: int
            Number of channels at the shallowest U-Net level. Depth doubles channels.
        depth: int
            Number of U-Net resolution levels (1 stem + (depth-1) downsampling levels).
        time_embed_dim: int
            Dimensionality of the sinusoidal flow-time embedding before its MLP projection.
        cbam_reduction: int
            Channel reduction ratio used inside each CBAM's channel-attention MLP.
        mean, std:
            Latent-space normalization statistics, mirroring
            `CuboidTransformerUNet`'s `normalize`/`unnormalize` contract used by the
            training loop. Stored as plain attributes (not buffers/parameters),
            matching `CuboidTransformerUNet` exactly: they stay on CPU even after
            `.to(device)`, which the reused `partial_evaluate_model` (and PyTorch's
            0-dim-CPU-tensor/CUDA-tensor broadcasting) relies on -- e.g. it calls
            `model.std.numpy()` directly, which would break if these were CUDA buffers.
        """
        super().__init__()
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        assert depth >= 2, "depth must allow at least one downsample/upsample pair"
        t_in, h, w, c_in = input_shape
        t_out, h_out, w_out, c_out = target_shape
        assert (h, w) == (h_out, w_out), (
            f"spatial dims of cond {(h, w)} and x {(h_out, w_out)} must match"
        )
        self.in_len = t_in
        self.out_len = t_out
        self.latent_channels = c_out
        self.spatial_shape = (h, w)

        in_channels = t_in * c_in + t_out * c_out
        out_channels = t_out * c_out

        channels = [base_channels * (2**i) for i in range(depth)]

        time_hidden_dim = time_embed_dim * 4
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_hidden_dim),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, time_hidden_dim),
        )

        self.stem = DoubleDSConv(in_channels, channels[0])
        self.film_stem = FiLM(time_hidden_dim, channels[0])
        self.cbam_stem = CBAM(channels[0], reduction=cbam_reduction)

        self.downs = nn.ModuleList(
            [Down(channels[i], channels[i + 1]) for i in range(depth - 1)]
        )
        self.film_downs = nn.ModuleList(
            [FiLM(time_hidden_dim, channels[i + 1]) for i in range(depth - 1)]
        )
        self.cbam_downs = nn.ModuleList(
            [CBAM(channels[i + 1], reduction=cbam_reduction) for i in range(depth - 1)]
        )

        self.ups = nn.ModuleList(
            [
                Up(channels[i + 1], channels[i], channels[i])
                for i in range(depth - 2, -1, -1)
            ]
        )
        self.film_ups = nn.ModuleList(
            [FiLM(time_hidden_dim, channels[i]) for i in range(depth - 2, -1, -1)]
        )

        self.final_proj = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean

    def forward(self, t, x, cond, verbose=False):
        """
        Parameters
        ----------
        t:  torch.Tensor, shape (B,)
        x:  torch.Tensor, shape (B, T_out, H, W, C)
        cond:   torch.Tensor, shape (B, T_in, H, W, C)
        verbose:    bool

        Returns
        -------
        out:    torch.Tensor, shape (B, T_out, H, W, C)
        """
        cond_flat = rearrange(cond, "b t h w c -> b (t c) h w")
        x_flat = rearrange(x, "b t h w c -> b (t c) h w")
        h = torch.cat([cond_flat, x_flat], dim=1)

        t_emb = self.time_mlp(timestep_embedding(t, self.time_embed_dim))

        h = self.stem(h)
        h = self.film_stem(h, t_emb)
        h = self.cbam_stem(h)
        skips = [h]
        if verbose:
            print(f"stem out: {h.shape}")

        for down, film, cbam in zip(self.downs, self.film_downs, self.cbam_downs):
            h = down(h)
            h = film(h, t_emb)
            h = cbam(h)
            skips.append(h)
            if verbose:
                print(f"down out: {h.shape}")

        skips.pop()  # deepest feature map is the bottleneck `h` itself, not a skip
        for up, film in zip(self.ups, self.film_ups):
            skip = skips.pop()
            h = up(h, skip)
            h = film(h, t_emb)
            if verbose:
                print(f"up out: {h.shape}")

        out = self.final_proj(h)
        out = rearrange(
            out, "b (t c) h w -> b t h w c", t=self.out_len, c=self.latent_channels
        )
        return out
