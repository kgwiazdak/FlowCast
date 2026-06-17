"""Tests for the deterministic regression ablation mode (issue #4).

Verifies: the cfm/deterministic training-mode switch uses the same backbone
and shape in both modes; deterministic mode skips noise/time sampling in
favor of a zero placeholder + fixed t=1; deterministic partial evaluation
uses a single direct forward pass (no ODE integration, no ensemble); and a
warning is emitted when test_params.probabilistic_samples > 1 while running
in deterministic mode.
"""

import warnings

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.sevir.runner.smaat_cfm.train_smaat_cfm import (
    run_training,
    compute_model_output_and_target,
    partial_evaluate_model_deterministic,
)
from common.cfm.cfm import ConditionalFlowMatcher
from common.models.smaat_cfm.backbone import SmaatCFMBackbone

from conftest import T_IN, T_OUT, H, W, C, SyntheticLatentDataset, make_config


def test_deterministic_mode_uses_zero_placeholder_and_t_one():
    model = SmaatCFMBackbone((T_IN, H, W, C), (T_OUT, H, W, C), depth=3)
    flow_matcher = ConditionalFlowMatcher(sigma=0.01)
    x0_cond = torch.randn(2, T_IN, H, W, C)
    x1 = torch.randn(2, T_OUT, H, W, C)

    captured = {}
    real_forward = model.forward

    def spy_forward(t, x, cond, verbose=False):
        captured["t"] = t.clone()
        captured["x"] = x.clone()
        return real_forward(t, x, cond, verbose=verbose)

    model.forward = spy_forward
    pred, target = compute_model_output_and_target(model, flow_matcher, x0_cond, x1, "deterministic", torch.device("cpu"))

    assert torch.equal(captured["t"], torch.ones(2))
    assert torch.equal(captured["x"], torch.zeros(2, T_OUT, H, W, C))
    assert torch.equal(target, x1)
    assert pred.shape == x1.shape


def test_cfm_mode_samples_noise_and_predicts_velocity():
    model = SmaatCFMBackbone((T_IN, H, W, C), (T_OUT, H, W, C), depth=3)
    flow_matcher = ConditionalFlowMatcher(sigma=0.01)
    x0_cond = torch.randn(2, T_IN, H, W, C)
    x1 = torch.randn(2, T_OUT, H, W, C)

    pred, target = compute_model_output_and_target(model, flow_matcher, x0_cond, x1, "cfm", torch.device("cpu"))
    assert pred.shape == x1.shape
    assert target.shape == x1.shape


def test_unknown_training_mode_raises():
    model = SmaatCFMBackbone((T_IN, H, W, C), (T_OUT, H, W, C), depth=3)
    flow_matcher = ConditionalFlowMatcher(sigma=0.01)
    x0_cond = torch.randn(2, T_IN, H, W, C)
    x1 = torch.randn(2, T_OUT, H, W, C)
    try:
        compute_model_output_and_target(model, flow_matcher, x0_cond, x1, "not_a_real_mode", torch.device("cpu"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_training_smoke_deterministic_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = make_config(tmp_path, "smaat", training_mode="deterministic")
    train_dataset = SyntheticLatentDataset(num_samples=12, seed=0)
    val_dataset = SyntheticLatentDataset(num_samples=6, seed=1)

    result = run_training(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=torch.device("cpu"),
        run_id="smoke_deterministic",
    )
    assert result["global_step"] > 0


def test_deterministic_mode_warns_on_probabilistic_samples_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = make_config(tmp_path, "smaat", training_mode="deterministic", probabilistic_samples=8)
    train_dataset = SyntheticLatentDataset(num_samples=4, seed=0)
    val_dataset = SyntheticLatentDataset(num_samples=4, seed=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=torch.device("cpu"),
            run_id="smoke_warn",
        )
    assert any("probabilistic_samples" in str(w.message) for w in caught)


def test_deterministic_mode_no_warning_when_probabilistic_samples_is_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = make_config(tmp_path, "smaat", training_mode="deterministic", probabilistic_samples=1)
    train_dataset = SyntheticLatentDataset(num_samples=4, seed=0)
    val_dataset = SyntheticLatentDataset(num_samples=4, seed=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=torch.device("cpu"),
            run_id="smoke_nowarn",
        )
    assert not any("probabilistic_samples" in str(w.message) for w in caught)


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


def test_partial_evaluate_model_deterministic_runs_without_ae(tmp_path):
    """Exercises partial_evaluate_model_deterministic's full plumbing (no ODE
    integration, single direct forward pass) with ae_model=None, which falls
    back to a random-latent encode/decode path -- enough to confirm the
    function runs end-to-end and returns sane metric keys without requiring a
    real pretrained VAE (out of scope for this unit test; see issue #5)."""
    model = SmaatCFMBackbone((T_IN, PIXEL_H // 8, PIXEL_W // 8, 4), (T_OUT, PIXEL_H // 8, PIXEL_W // 8, 4), depth=2)
    model.eval()

    dataset = SyntheticPixelSampleDataset(num_samples=4, lag_time=T_IN, lead_time=T_OUT, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda batch: (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        [b[2] for b in batch],
    ))

    results = partial_evaluate_model_deterministic(
        model=model,
        device=torch.device("cpu"),
        val_sample_loader=loader,
        thresholds=__import__("numpy").array([16, 74, 133], dtype="float32"),
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
        ema_model_evaluated=False,
    )
    assert results is not None
    assert "mse_from_mean_mean" in results
    assert "csi_from_mean_m" in results
