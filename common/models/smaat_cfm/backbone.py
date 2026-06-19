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
original SmaAt-UNet's pure-2D, early-fusion philosophy (it stacks its own multi-frame
input the same way). Flow time `t` is injected via a sinusoidal embedding + small MLP,
applied with AdaGN-FiLM (see layers.py) after the stem and after every encoder/decoder
block, matching DDPM/ADM's "condition at every level" practice.

Two deliberate deviations from the literal official SmaAt-UNet, both researched and
confirmed necessary/justified rather than incidental (see project plan):
  - BatchNorm -> GroupNorm throughout, plus the new AdaGN-FiLM conditioning module:
    the original architecture has no timestep conditioning at all, since it's a
    one-shot deterministic regressor, not a generative model component.
  - Downsampling depth reduced from the original's 4 stages (tuned for 288x288 raw
    pixel maps) to 2 stages on this latent's 48x48 grid -- a 4-stage literal port
    would bottleneck at 3x3, well below the ~8x8-12x12 floor used by comparable
    latent-space conv/diffusion U-Nets (e.g. Stable Diffusion's 64x64->8x8) and by
    FlowCast's own Earthformer-UNet backbone (2 hierarchical stages on this same grid).
Channel widths (64 base, doubling per stage, //2 "bilinear factor" reduction at the
bottleneck and at every decoder stage above the stem) are kept literally identical to
the official repo's convention.
"""

import torch
from einops import rearrange
from torch import nn

from common.models.flowcast.utils import timestep_embedding
from common.models.smaat_cfm.layers import CBAM, AdaGNFiLM, Down, DoubleConvDS, Up

_BILINEAR_FACTOR = 2  # this implementation always upsamples via bilinear interpolation


class SmaatCFMBackbone(nn.Module):
    def __init__(
        self,
        input_shape,
        target_shape,
        base_channels=64,
        depth=3,
        time_embed_dim=128,
        cbam_reduction=16,
        kernels_per_layer=2,
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
            Channel count at the stem (shallowest U-Net level). 64 matches the
            official SmaAt-UNet's first-layer width.
        depth: int
            Number of U-Net resolution levels (1 stem + (depth-1) downsampling
            levels). Default 3 = 2 downsamples, bottleneck at H/4 x W/4 (12x12 for
            this project's 48x48 latents) -- see module docstring for why this is
            shallower than the official repo's depth=5 (4 downsamples).
        time_embed_dim: int
            Dimensionality of the sinusoidal flow-time embedding before its MLP projection.
        cbam_reduction: int
            Channel reduction ratio used inside each CBAM's channel-attention MLP.
        kernels_per_layer: int
            Depthwise depth multiplier, matching the official SmaAt-UNet's own term
            and its published default of 2 (not the library default of 1).
        mean, std:
            Latent-space normalization statistics, mirroring
            `CuboidTransformerUNet`'s `normalize`/`unnormalize` contract used by the
            training loop. Stored as plain attributes (not buffers/parameters),
            matching `CuboidTransformerUNet` exactly: they stay on CPU even after
            `.to(device)`, which the reused `partial_evaluate_model` relies on (it
            calls `model.std.numpy()` directly).
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

        # Encoder channel widths, matching the official repo's doubling-per-stage
        # convention, with the "bilinear factor" halving applied at the bottleneck
        # (official: 1024 -> 1024 // factor = 512 when bilinear=True, since the
        # parameter-free bilinear upsample can't "earn" the extra capacity that a
        # learned transposed-conv upsample would).
        channels = [base_channels * (2**i) for i in range(depth)]
        channels[-1] = channels[-1] // _BILINEAR_FACTOR

        time_hidden_dim = time_embed_dim * 4
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_hidden_dim),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, time_hidden_dim),
        )

        self.stem = DoubleConvDS(in_channels, channels[0], kernels_per_layer=kernels_per_layer)
        self.film_stem = AdaGNFiLM(time_hidden_dim, channels[0])
        self.cbam_stem = CBAM(channels[0], reduction=cbam_reduction)

        self.downs = nn.ModuleList(
            [Down(channels[i], channels[i + 1], kernels_per_layer=kernels_per_layer) for i in range(depth - 1)]
        )
        self.film_downs = nn.ModuleList(
            [AdaGNFiLM(time_hidden_dim, channels[i + 1]) for i in range(depth - 1)]
        )
        self.cbam_downs = nn.ModuleList(
            [CBAM(channels[i + 1], reduction=cbam_reduction) for i in range(depth - 1)]
        )

        # Decoder output widths mirror the official repo: each stage above the stem
        # halves the corresponding encoder skip's channel count (again the
        # "bilinear factor" trick); the final (stem-level) stage outputs the base
        # width unchanged, matching the official repo's last `up` block. Each `Up`
        # receives the *previous* stage's actual output as its low-res input (the
        # bottleneck for the first/deepest up, then each prior up's own output) --
        # not the raw encoder channel count, which only coincides with it once.
        up_out_channels = [channels[i] if i == 0 else channels[i] // _BILINEAR_FACTOR for i in range(depth - 1)]
        ups, films = [], []
        low_res_channels = channels[-1]  # bottleneck output feeds the first/deepest up
        for i in range(depth - 2, -1, -1):
            ups.append(Up(low_res_channels, channels[i], up_out_channels[i], kernels_per_layer=kernels_per_layer))
            films.append(AdaGNFiLM(time_hidden_dim, up_out_channels[i]))
            low_res_channels = up_out_channels[i]
        self.ups = nn.ModuleList(ups)
        self.film_ups = nn.ModuleList(films)
        # No CBAM in the decoder: the official SmaAt-UNet only attends encoder
        # features (CBAM-attended encoder outputs feed the decoder as skips).

        self.final_proj = nn.Conv2d(up_out_channels[0], out_channels, kernel_size=1)

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
