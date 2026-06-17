"""Shared synthetic fixtures for the single-GPU training loop tests
(tests/test_train_smaat_cfm_single_gpu.py and tests/test_train_smaat_cfm_deterministic.py).

No real SEVIR HDF5 files, no VAE, no GPU required.
"""

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

T_IN, T_OUT, H, W, C = 4, 3, 8, 8, 4


class SyntheticLatentDataset(Dataset):
    """Mimics DynamicEncodedSequentialSevirDataset's (X, Y, metadata) contract
    with small, fully synthetic latent-shaped tensors."""

    def __init__(self, num_samples, seed):
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(num_samples, T_IN, H, W, C, generator=gen)
        self.y = torch.randn(num_samples, T_OUT, H, W, C, generator=gen)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], {"id": idx}


def make_config(
    tmp_path,
    backbone_type,
    training_mode="cfm",
    probabilistic_samples=None,
    early_stopping_metric="val_loss",
    partial_evaluation=False,
):
    config = OmegaConf.create(
        {
            "smaat_model": {"base_channels": 8, "depth": 3, "time_embed_dim": 16, "cbam_reduction": 4},
            "latent_model": {
                "base_units": 8, "scale_alpha": 1.0, "num_heads": 2, "attn_drop": 0.0, "proj_drop": 0.0,
                "ffn_drop": 0.0, "downsample": 2, "downsample_type": "patch_merge", "upsample_type": "upsample",
                "upsample_kernel_size": 3, "depth": [1, 1], "self_pattern": "axial", "num_global_vectors": 0,
                "use_global_vector_ffn": False, "use_global_self_attn": False, "separate_global_qkv": False,
                "global_dim_ratio": 1, "ffn_activation": "leaky", "gated_ffn": False, "norm_layer": "layer_norm",
                "padding_type": "ignore", "checkpoint_level": 0, "pos_embed_type": "t+h+w",
                "use_relative_pos": True, "self_attn_use_final_proj": True, "attn_linear_init_mode": "0",
                "ffn_linear_init_mode": "0", "ffn2_linear_init_mode": "2", "attn_proj_linear_init_mode": "2",
                "conv_init_mode": "0", "down_up_linear_init_mode": "0", "global_proj_linear_init_mode": "2",
                "norm_init_mode": "0", "time_embed_channels_mult": 4, "time_embed_use_scale_shift_norm": False,
                "time_embed_dropout": 0.0, "unet_res_connect": True,
            },
            "training_params": {
                "micro_batch_size": 2, "num_epochs": 2, "num_workers": 0, "early_stopping_patience": 10,
                "early_stopping_metric": early_stopping_metric, "grad_accumulation_steps": 1, "gradient_clip_val": 1.0,
                "fp16": False,
            },
            "optimizer_params": {"learning_rate": 1e-3, "optimizer_type": "adamw", "weight_decay": 0.0},
            "scheduler_params": {
                "scheduler_type": "cosine", "lr_plateau_factor": 0.2, "lr_plateau_patience": 2,
                "lr_cosine_warmup_iter_percentage": 0.1, "lr_cosine_min_warmup_lr_ratio": 0.1,
                "lr_cosine_min_lr_ratio": 0.1,
            },
            "data_params": {"lag_time": T_IN, "lead_time": T_OUT, "time_spacing": 1},
            "run_params": {
                "debug_mode": False, "enable_wandb": False, "backbone_type": backbone_type,
                "training_mode": training_mode, "run_string": "test", "preload_model": None,
            },
            "flow_matching_params": {"flow_matching_method": "vanilla", "sigma": 0.01},
            "partial_evaluation_params": {
                "partial_evaluation": partial_evaluation, "partial_evaluation_interval": 1,
                "partial_evaluation_batches": 1, "cartopy_features": False,
            },
            "autoencoder_params": {"normalized_autoencoder": False},
            "ema_model_saving_params": {"ema_model_saving": True, "ema_model_saving_decay": 0.999},
        }
    )
    if probabilistic_samples is not None:
        config.test_params = {"probabilistic_samples": probabilistic_samples}
    return config
