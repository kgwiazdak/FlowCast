"""Tests for train_smaat_cfm_deterministic.py.

This script is a diagnostic ablation, not the primary experiment (see its
module docstring): it skips CFM's noise/time sampling and asks the backbone to
directly regress the future latent from the past conditioning, isolating
whether the backbone has any learnable capacity at all from whether the CFM
framework works with it. These tests verify: the script does not depend on
the CFM/ODE machinery at all (no flow matcher, no odeint), the CLI default
points at the shared SmaAt config, and partial_evaluate_model_deterministic
itself runs end-to-end (single direct forward pass, no ODE integration, no
ensemble) and returns sane metric keys.
"""

import inspect

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import experiments.sevir.runner.smaat_cfm.train_smaat_cfm_deterministic as train_det
from common.models.smaat_cfm.backbone import SmaatCFMBackbone

from conftest import T_IN, T_OUT


def test_config_default_points_at_smaat_config():
    args = train_det.parser.parse_args([])
    assert args.config == "experiments/sevir/runner/smaat_cfm/smaat_cfm_config.yaml"


def test_script_does_not_use_cfm_or_ode_machinery():
    """A deterministic ablation has no flow time to sample and nothing to
    integrate -- verify the CFM/ODE imports were dropped, not just unused."""
    source = inspect.getsource(train_det)
    assert "ConditionalFlowMatcher" not in source
    assert "odeint" not in source
    assert "SmaatCFMBackbone(" in source


PIXEL_H, PIXEL_W = 32, 32


class SyntheticPixelSampleDataset(Dataset):
    """Mimics DynamicSequentialSevirDataset's (X, Y, metadata) contract: pixel-space
    (not latent) tensors shaped (C=1, T, H, W), as consumed by partial evaluation."""

    def __init__(self, num_samples, lag_time, lead_time, seed):
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.rand(num_samples, 1, lag_time, PIXEL_H, PIXEL_W, generator=gen) * 255.0
        self.y = torch.rand(num_samples, 1, lead_time, PIXEL_H, PIXEL_W, generator=gen) * 255.0

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], {"id": idx}


def test_partial_evaluate_model_deterministic_runs_without_ae():
    """Exercises partial_evaluate_model_deterministic's full plumbing (no ODE
    integration, single direct forward pass) with ae_model=None, which falls
    back to a random-latent encode/decode path -- enough to confirm the
    function runs end-to-end and returns sane metric keys without requiring a
    real pretrained VAE."""
    model = SmaatCFMBackbone((T_IN, PIXEL_H // 8, PIXEL_W // 8, 4), (T_OUT, PIXEL_H // 8, PIXEL_W // 8, 4), depth=2)
    model.eval()

    dataset = SyntheticPixelSampleDataset(num_samples=4, lag_time=T_IN, lead_time=T_OUT, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda batch: (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        [b[2] for b in batch],
    ))

    results = train_det.partial_evaluate_model_deterministic(
        model=model,
        device=torch.device("cpu"),
        val_sample_loader=loader,
        thresholds=np.array([16, 74, 133], dtype="float32"),
        global_step=0,
        epoch=0,
        ae_model=None,
        normalized_autoencoder=False,
        use_fp16=False,
        partial_evaluation_batches=2,
        lead_time=T_OUT,
        enable_wandb=False,
        wandb_instance=None,
        debug_print_prefix="[test] ",
        plots_folder="artifacts/test_tmp/plots",
        cartopy_features=False,
        ema_model_evaluated=False,
    )
    assert results is not None
    assert "mse_from_mean_mean" in results
    assert "csi_from_mean_m" in results
