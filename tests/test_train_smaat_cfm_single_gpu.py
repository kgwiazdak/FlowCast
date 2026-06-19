"""Tests for train_smaat_cfm.py.

This script is a literal copy of dist_train_flowcast.py with exactly one
change: the model construction block builds SmaatCFMBackbone instead of
CuboidTransformerUNet. There is no monkeypatching, factory, or config-mode
machinery left to unit-test in isolation -- the full training loop requires
real SEVIR data and DDP setup, exercised separately, not here. These tests
cover what is mechanically verifiable without that: the module imports the
right backbone class, the CLI default points at the SmaAt config, and the
small pure-Python helpers (setup_ddp's non-distributed fallback, cleanup_ddp's
no-op when DDP was never initialized) behave correctly.
"""

import inspect

import torch

import experiments.sevir.runner.smaat_cfm.train_smaat_cfm as train_smaat_cfm


def test_config_default_points_at_smaat_config():
    args = train_smaat_cfm.parser.parse_args([])
    assert args.config == "experiments/sevir/runner/smaat_cfm/smaat_cfm_config.yaml"


def test_model_construction_uses_smaat_backbone_not_cuboid():
    """A literal copy must swap exactly the model class -- verify the source
    references SmaatCFMBackbone and does not import CuboidTransformerUNet."""
    source = inspect.getsource(train_smaat_cfm)
    assert "SmaatCFMBackbone(" in source
    assert "import CuboidTransformerUNet" not in source
    assert "CuboidTransformerUNet(" not in source


def test_setup_ddp_falls_back_to_single_device_without_rank_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    rank, world_size, local_rank, device = train_smaat_cfm.setup_ddp()
    assert (rank, world_size, local_rank) == (0, 1, 0)
    assert device.type in ("cuda", "cpu")


def test_cleanup_ddp_is_a_noop_when_not_initialized():
    import torch.distributed as dist

    assert not dist.is_initialized()
    train_smaat_cfm.cleanup_ddp()  # must not raise
