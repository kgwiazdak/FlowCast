"""Level-1 contract tests for the SmaAt-CFM backbone.

These tests run entirely on synthetic tensors -- no SEVIR dataset, no GPU, no
pretrained VAE -- and verify the external behavior of the forward(t, x, cond)
contract rather than internal layer details: shapes, gradient flow, parameter
count, sensitivity to the flow-time conditioning, and convergence on a tiny
fixed batch. This is the first verification gate before using real data.
"""

import torch

from common.models.smaat_cfm.backbone import SmaatCFMBackbone

INPUT_SHAPE = (13, 24, 24, 4)  # (T_in, H, W, C) -- matches SEVIR VIL latent dims
TARGET_SHAPE = (12, 24, 24, 4)  # (T_out, H, W, C)
BATCH_SIZE = 2

MIN_EXPECTED_PARAMS = 1_000_000
MAX_EXPECTED_PARAMS = 10_000_000


def make_model():
    return SmaatCFMBackbone(INPUT_SHAPE, TARGET_SHAPE)


def make_batch(batch_size=BATCH_SIZE, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, *TARGET_SHAPE, generator=gen)
    cond = torch.randn(batch_size, *INPUT_SHAPE, generator=gen)
    t = torch.rand(batch_size, generator=gen)
    return t, x, cond


def test_output_shape_matches_target():
    model = make_model()
    t, x, cond = make_batch()
    out = model(t, x, cond)
    assert out.shape == x.shape


def test_output_is_finite():
    model = make_model()
    t, x, cond = make_batch()
    out = model(t, x, cond)
    assert torch.isfinite(out).all()


def test_gradient_flows_to_all_parameters():
    model = make_model()
    t, x, cond = make_batch()
    out = model(t, x, cond)
    out.sum().backward()
    missing_grad = [
        name for name, p in model.named_parameters() if p.grad is None
    ]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"


def test_parameter_count_in_expected_range():
    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SmaAt-CFM backbone parameter count: {n_params} ({n_params / 1e6:.2f}M)")
    assert MIN_EXPECTED_PARAMS <= n_params <= MAX_EXPECTED_PARAMS


def test_normalize_unnormalize_roundtrip():
    """normalize/unnormalize must mirror CuboidTransformerUNet's contract, since the
    training loop calls them directly on the model (outside of forward())."""
    model = SmaatCFMBackbone(INPUT_SHAPE, TARGET_SHAPE, mean=2.0, std=3.0)
    x = torch.randn(BATCH_SIZE, *TARGET_SHAPE)
    normalized = model.normalize(x)
    assert torch.allclose(normalized, (x - 2.0) / 3.0)
    assert torch.allclose(model.unnormalize(normalized), x, atol=1e-5)


def test_output_sensitive_to_flow_time():
    model = make_model()
    model.eval()
    _, x, cond = make_batch()
    batch_size = x.shape[0]
    t_low = torch.zeros(batch_size)
    t_high = torch.ones(batch_size)
    with torch.no_grad():
        out_low = model(t_low, x, cond)
        out_high = model(t_high, x, cond)
    assert not torch.allclose(out_low, out_high, atol=1e-5)


def test_loss_converges_on_small_synthetic_batch():
    """Convergence is checked on a smaller spatial/temporal extent than the other
    contract tests (purely for optimization speed/robustness in CI -- shape
    correctness at full SEVIR-like dims is already covered above) with a wider
    batch to keep BatchNorm statistics stable.
    """
    torch.manual_seed(0)
    small_input_shape = (4, 16, 16, 4)
    small_target_shape = (3, 16, 16, 4)
    model = SmaatCFMBackbone(small_input_shape, small_target_shape)

    gen = torch.Generator().manual_seed(1)
    batch_size = 8
    x = torch.randn(batch_size, *small_target_shape, generator=gen)
    cond = torch.randn(batch_size, *small_input_shape, generator=gen)
    t = torch.rand(batch_size, generator=gen)
    target_velocity = torch.randn(batch_size, *small_target_shape, generator=gen)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    def step():
        optimizer.zero_grad()
        pred = model(t, x, cond)
        loss = torch.nn.functional.mse_loss(pred, target_velocity)
        loss.backward()
        optimizer.step()
        return loss.item()

    initial_loss = step()
    final_loss = initial_loss
    for _ in range(110):
        final_loss = step()

    assert final_loss < initial_loss * 0.5, (
        f"loss did not drop by at least half: initial={initial_loss:.4f}, "
        f"final={final_loss:.4f}"
    )
