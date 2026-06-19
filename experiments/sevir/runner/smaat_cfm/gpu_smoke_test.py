"""Stage 0+1 GPU smoke test for the SmaAt-CFM backbone -- no SEVIR data required.

Run this FIRST on a fresh GPU machine, before touching real data. It exercises the
exact same computation graph as `train_smaat_cfm.py`'s training loop (normalize,
CFM noise/time sampling, forward, AMP autocast, backward, grad clip, optimizer
step, EMA update) and the deterministic ablation's substitution, on synthetic
tensors but on real CUDA hardware with fp16 autocast enabled -- the one thing the
CPU-only smoke tests from development couldn't verify empirically (whether the
backbone's plain, non-buffer `mean`/`std` attributes really do broadcast safely
against CUDA tensors, as reasoned on paper, rather than raising a device-mismatch
error in practice).

Usage:
    python experiments/sevir/runner/smaat_cfm/gpu_smoke_test.py

Exits non-zero on any failure, with a clear PASS/FAIL banner per stage.
"""

import copy
import sys

import torch
import torch.nn as nn

from common.models.smaat_cfm.backbone import SmaatCFMBackbone
from common.cfm.cfm import ConditionalFlowMatcher
from common.utils.utils import ema


def banner(ok, label):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


def main():
    failures = []

    print("=== Stage 0: environment ===")
    cuda_ok = torch.cuda.is_available()
    banner(cuda_ok, f"CUDA available (device: {torch.cuda.get_device_name(0) if cuda_ok else 'N/A'})")
    if not cuda_ok:
        print("No CUDA device visible -- aborting, fix the environment before continuing.")
        sys.exit(1)
    device = torch.device("cuda")

    try:
        import torchdiffeq  # noqa: F401
        import diffusers  # noqa: F401
        import omegaconf  # noqa: F401
        import wandb  # noqa: F401
        banner(True, "Required packages import cleanly (torchdiffeq, diffusers, omegaconf, wandb)")
    except ImportError as e:
        banner(False, f"Missing dependency: {e}")
        failures.append("imports")

    # Matches smaat_cfm_config.yaml's smaat_model defaults and the real SEVIR shapes.
    input_shape = (13, 48, 48, 4)
    target_shape = (12, 48, 48, 4)
    B = 3  # matches training_params.micro_batch_size

    print("\n=== Stage 1a: CFM-mode training step on GPU (fp16 autocast) ===")
    try:
        model = SmaatCFMBackbone(
            input_shape=input_shape, target_shape=target_shape,
            base_channels=64, depth=3, time_embed_dim=128, cbam_reduction=16, kernels_per_layer=2,
            mean=0.0, std=1.0,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  param count: {n_params} ({n_params / 1e6:.2f}M)")
        ema_model = copy.deepcopy(model).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = nn.MSELoss(reduction="none")
        flow_matcher = ConditionalFlowMatcher(sigma=0.01)
        scaler = torch.amp.GradScaler(device="cuda", enabled=True)

        torch.manual_seed(0)
        losses = []
        for step in range(5):
            x1 = torch.randn(B, *target_shape, device=device)
            x0_cond = torch.randn(B, *input_shape, device=device)
            x0_noise = torch.randn_like(x1, device=device)

            x0_cond = model.normalize(x0_cond)  # <-- the device-placement question under test
            x1 = model.normalize(x1)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=True):
                t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(x0_noise, x1)
                v_t = model(t, x_t, x0_cond)
                raw_loss = criterion(v_t, u_t)
                loss = raw_loss.mean(dim=list(range(1, raw_loss.ndim))).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema(model, ema_model, 0.999)

            losses.append(loss.item())
            print(f"  step {step}: loss={loss.item():.6f}")

        finite = all(torch.isfinite(torch.tensor(losses)))
        banner(finite, "CFM-mode: 5 steps ran on GPU with finite losses, normalize() did not raise a device error")
        if not finite:
            failures.append("cfm-mode-finite-loss")
    except Exception as e:
        banner(False, f"CFM-mode GPU step raised: {type(e).__name__}: {e}")
        failures.append("cfm-mode-exception")

    print("\n=== Stage 1b: deterministic-mode training step on GPU (fp16 autocast) ===")
    try:
        model = SmaatCFMBackbone(
            input_shape=input_shape, target_shape=target_shape,
            base_channels=64, depth=3, time_embed_dim=128, cbam_reduction=16, kernels_per_layer=2,
            mean=0.0, std=1.0,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = nn.MSELoss(reduction="none")
        scaler = torch.amp.GradScaler(device="cuda", enabled=True)

        torch.manual_seed(0)
        losses = []
        for step in range(5):
            x1 = torch.randn(B, *target_shape, device=device)
            x0_cond = torch.randn(B, *input_shape, device=device)
            x0_cond = model.normalize(x0_cond)
            x1 = model.normalize(x1)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=True):
                x_placeholder = torch.zeros_like(x1)
                t_full = torch.ones(x1.shape[0], device=device)
                pred = model(t_full, x_placeholder, x0_cond)
                raw_loss = criterion(pred, x1)
                loss = raw_loss.mean(dim=list(range(1, raw_loss.ndim))).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            print(f"  step {step}: loss={loss.item():.6f}")

        finite = all(torch.isfinite(torch.tensor(losses)))
        banner(finite, "Deterministic-mode: 5 steps ran on GPU with finite losses")
        if not finite:
            failures.append("deterministic-mode-finite-loss")
    except Exception as e:
        banner(False, f"Deterministic-mode GPU step raised: {type(e).__name__}: {e}")
        failures.append("deterministic-mode-exception")

    print("\n=== Summary ===")
    if failures:
        print(f"FAILED stages: {failures}")
        sys.exit(1)
    else:
        print("All GPU smoke checks passed. Safe to proceed to Stage 2 (real data, debug_mode).")


if __name__ == "__main__":
    main()
