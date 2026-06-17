"""
Single-GPU training script for Conditional Flow Matching on SEVIR, with a
config-driven choice of backbone: the original CuboidTransformerUNet or the
new SmaAt-CFM backbone.

Unlike `experiments/sevir/runner/flowcast/dist_train_flowcast.py`, this script
has no torchrun/DistributedDataParallel dependency -- it targets a single GPU
(or CPU, for smoke testing). Everything else is reused unchanged: the
Conditional Flow Matching framework, the frozen pretrained VAE, the static
latent-dataset mechanism, and the partial-evaluation/metrics logic (imported
directly from the original script, not duplicated).
"""

import sys
import os
import copy
import argparse
import datetime
import random
import warnings

import numpy as np
import wandb
import namegenerator
from tqdm import tqdm

# tqdm's background monitor thread has been observed to race with PyTorch's
# autograd engine threads on Windows, crashing the process with an access
# violation during backward(). Harmless to disable.
tqdm.monitor_interval = 0

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from omegaconf import OmegaConf

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments.sevir.dataset.sevirfulldataset import (
    DynamicEncodedSequentialSevirDataset,
    dynamic_encoded_sequential_collate,
    DynamicSequentialSevirDataset,
    dynamic_sequential_collate,
)
from common.utils.utils import EarlyStopping, compute_mean_std, warmup_lambda, ema
from common.models.flowcast.cuboid_transformer_unet import CuboidTransformerUNet
from common.models.smaat_cfm.backbone import SmaatCFMBackbone
from common.cfm.cfm import ConditionalFlowMatcher
from common.metrics.metrics_streaming_probabilistic import MetricsAccumulator
from common.utils.utils import calculate_metrics
from experiments.sevir.dataset.sevirfulldataset import post_process_samples
from experiments.sevir.runner.flowcast.dist_train_flowcast import (
    partial_evaluate_model,
)

THRESHOLDS = np.array([16, 74, 133, 160, 181, 219], dtype=np.float32)

parser = argparse.ArgumentParser(description="Single-GPU SmaAt-CFM / CuboidTransformerUNet training.")
parser.add_argument(
    "--config",
    type=str,
    default="experiments/sevir/runner/smaat_cfm/smaat_cfm_config.yaml",
)
parser.add_argument(
    "--train_file",
    type=str,
    default="datasets/sevir/data/sevir_full_latent_vae_kl1e4/nowcast_training_full.h5",
)
parser.add_argument(
    "--train_meta",
    type=str,
    default="datasets/sevir/data/sevir_full_latent_vae_kl1e4/nowcast_training_full_META.csv",
)
parser.add_argument(
    "--val_file",
    type=str,
    default="datasets/sevir/data/sevir_full_latent_vae_kl1e4/nowcast_validation_full.h5",
)
parser.add_argument(
    "--val_meta",
    type=str,
    default="datasets/sevir/data/sevir_full_latent_vae_kl1e4/nowcast_validation_full_META.csv",
)
parser.add_argument(
    "--partial_evaluation_file",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_validation_full.h5",
)
parser.add_argument(
    "--partial_evaluation_meta",
    type=str,
    default="datasets/sevir/data/sevir_full/nowcast_validation_full_META.csv",
)


def build_backbone(backbone_type, input_shape, target_shape, config, mean, std):
    """Instantiates either the original CuboidTransformerUNet or the new
    SmaAt-CFM backbone, both sharing the same forward(t, x, cond) contract."""
    if backbone_type == "cuboid":
        model_config = OmegaConf.to_object(config.latent_model)
        return CuboidTransformerUNet(
            input_shape=input_shape,
            target_shape=target_shape,
            base_units=model_config["base_units"],
            block_units=None,
            scale_alpha=model_config["scale_alpha"],
            num_heads=model_config["num_heads"],
            attn_drop=model_config["attn_drop"],
            proj_drop=model_config["proj_drop"],
            ffn_drop=model_config["ffn_drop"],
            downsample=model_config["downsample"],
            downsample_type=model_config["downsample_type"],
            upsample_type=model_config["upsample_type"],
            upsample_kernel_size=model_config["upsample_kernel_size"],
            depth=model_config["depth"],
            block_attn_patterns=[model_config["self_pattern"]] * len(model_config["depth"]),
            num_global_vectors=model_config["num_global_vectors"],
            use_global_vector_ffn=model_config["use_global_vector_ffn"],
            use_global_self_attn=model_config["use_global_self_attn"],
            separate_global_qkv=model_config["separate_global_qkv"],
            global_dim_ratio=model_config["global_dim_ratio"],
            ffn_activation=model_config["ffn_activation"],
            gated_ffn=model_config["gated_ffn"],
            norm_layer=model_config["norm_layer"],
            padding_type=model_config["padding_type"],
            checkpoint_level=model_config["checkpoint_level"],
            pos_embed_type=model_config["pos_embed_type"],
            use_relative_pos=model_config["use_relative_pos"],
            self_attn_use_final_proj=model_config["self_attn_use_final_proj"],
            attn_linear_init_mode=model_config["attn_linear_init_mode"],
            ffn_linear_init_mode=model_config["ffn_linear_init_mode"],
            ffn2_linear_init_mode=model_config["ffn2_linear_init_mode"],
            attn_proj_linear_init_mode=model_config["attn_proj_linear_init_mode"],
            conv_init_mode=model_config["conv_init_mode"],
            down_linear_init_mode=model_config["down_up_linear_init_mode"],
            up_linear_init_mode=model_config["down_up_linear_init_mode"],
            global_proj_linear_init_mode=model_config["global_proj_linear_init_mode"],
            norm_init_mode=model_config["norm_init_mode"],
            time_embed_channels_mult=model_config["time_embed_channels_mult"],
            time_embed_use_scale_shift_norm=model_config["time_embed_use_scale_shift_norm"],
            time_embed_dropout=model_config["time_embed_dropout"],
            unet_res_connect=model_config["unet_res_connect"],
            mean=mean,
            std=std,
        )
    elif backbone_type == "smaat":
        smaat_config = OmegaConf.to_object(config.smaat_model)
        return SmaatCFMBackbone(
            input_shape=input_shape,
            target_shape=target_shape,
            base_channels=smaat_config["base_channels"],
            depth=smaat_config["depth"],
            time_embed_dim=smaat_config["time_embed_dim"],
            cbam_reduction=smaat_config["cbam_reduction"],
            mean=mean,
            std=std,
        )
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type!r}. Expected 'cuboid' or 'smaat'.")


def compute_model_output_and_target(model, flow_matcher, x0_cond, x1, training_mode, device):
    """Produces (model_output, regression_target) for one batch, branching on
    training_mode so the same backbone/shape is used in both modes:

    - "cfm": samples noise x0 and flow time t, interpolates x_t along the CFM
      path, and asks the model to predict the velocity field u_t = x1 - x0.
    - "deterministic": skips noise/time sampling entirely. The "noised future
      state" input is replaced by a constant zero placeholder and the flow
      time is fixed at t=1 (the "fully resolved" value), so the model directly
      regresses the (normalized) future latent state x1 from the past (cond)
      -- isolating the backbone's contribution from the CFM framework's.
    """
    if training_mode == "cfm":
        x0_noise = torch.randn_like(x1, device=device)
        t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(x0_noise, x1)
        v_t = model(t, x_t, x0_cond)
        return v_t, u_t
    elif training_mode == "deterministic":
        batch_size = x1.shape[0]
        x_placeholder = torch.zeros_like(x1)
        t_full = torch.ones(batch_size, device=device)
        pred = model(t_full, x_placeholder, x0_cond)
        return pred, x1
    else:
        raise ValueError(f"Unknown training_mode: {training_mode!r}. Expected 'cfm' or 'deterministic'.")


def partial_evaluate_model_deterministic(
    model,
    device,
    val_sample_loader,
    thresholds,
    global_step,
    epoch,
    ae_model,
    normalized_autoencoder,
    use_fp16,
    partial_evaluation_batches,
    lead_time,
    enable_wandb,
    wandb_instance,
    debug_print_prefix,
    ema_model_evaluated,
    batch_size_autoencoder=None,
):
    """Deterministic-mode counterpart to `partial_evaluate_model` (imported above).

    In CFM mode, the model output is a velocity field that has to be ODE-integrated
    from random noise to get a prediction. In deterministic mode the model directly
    outputs the predicted future latent state (see compute_model_output_and_target),
    so there is no ODE to integrate, no ensemble to sample, and only ever one
    prediction per input -- using `partial_evaluate_model`'s ODE-integration path
    here would be meaningless. The encode/normalize/decode/metrics plumbing mirrors
    `partial_evaluate_model`; only the prediction step differs.
    """
    results = None
    model.eval()
    if ae_model:
        ae_model.eval()

    with torch.no_grad():
        metrics_accumulators = [
            MetricsAccumulator(
                lead_time=lt,
                thresholds=thresholds,
                pool_size=16,
                compute_mse=True,
                compute_threshold=True,
                compute_crps=False,
                compute_fss=True,
                fss_scales=[1, 4, 16],
                device=device,
            )
            for lt in range(lead_time)
        ]
        count = 0
        y_pred_batches = []
        y_true_batches = []

        eval_bar = tqdm(val_sample_loader, desc=f"Partial Eval (deterministic) Epoch {epoch}", leave=False)
        for batch in eval_bar:
            x_cond, x_true, metadata = batch
            x_cond = x_cond.to(device, non_blocking=True)
            x_true = x_true.to(device, non_blocking=True)

            B, C, T_in, H, W = x_cond.shape
            x_cond = x_cond.permute(0, 2, 1, 3, 4).reshape(B * T_in, C, H, W)
            if normalized_autoencoder:
                x_cond = x_cond / 255.0

            if ae_model:
                encoded_chunks = []
                bs_ae = batch_size_autoencoder if batch_size_autoencoder is not None else x_cond.shape[0]
                for i in range(0, x_cond.shape[0], bs_ae):
                    chunk = x_cond[i : i + bs_ae]
                    encoded_chunk = ae_model.encode(chunk).latent_dist.mode()
                    encoded_chunks.append(encoded_chunk)
                x_cond = torch.cat(encoded_chunks, dim=0)
            else:
                print(f"{debug_print_prefix}Warning: AE model not available for encoding in partial eval.")
                latent_channels, latent_H, latent_W = 4, H // 8, W // 8
                x_cond = torch.randn(B * T_in, latent_channels, latent_H, latent_W, device=device)

            latent_channels, latent_H, latent_W = x_cond.shape[1], x_cond.shape[2], x_cond.shape[3]
            x_cond = x_cond.reshape(B, T_in, latent_channels, latent_H, latent_W).permute(0, 2, 1, 3, 4)
            x_cond = model.normalize(x_cond)
            x_cond = x_cond.permute(0, 2, 3, 4, 1)

            B, Tin, Hz, Wz, Cz = x_cond.shape
            x_true = x_true.squeeze(1)
            T_future = x_true.shape[1]
            H_true, W_true = x_true.shape[2], x_true.shape[3]

            x_placeholder = torch.zeros((B, T_future, Hz, Wz, Cz), device=device)
            t_full = torch.ones(B, device=device)
            with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
                x_pred_sample = model(t_full, x_placeholder, x_cond)

            x_pred = x_pred_sample.unsqueeze(1)  # single sample -> (B, S=1, T, Hz, Wz, Cz)

            x_pred_np = x_pred.cpu().numpy()
            x_true_np = x_true.cpu().numpy()
            x_pred_np = (x_pred_np * model.std.numpy() + model.mean.numpy()).astype(np.float32)

            B, S, T, H_latent, W_latent, C_latent = x_pred_np.shape
            x_pred_np = x_pred_np.reshape(B * S * T, H_latent, W_latent, C_latent)
            x_pred_tensor = torch.from_numpy(x_pred_np).to(device)
            x_pred_tensor = x_pred_tensor.permute(0, 3, 1, 2)

            if ae_model:
                decoded_chunks = []
                bs_ae = batch_size_autoencoder if batch_size_autoencoder is not None else x_pred_tensor.shape[0]
                for i in range(0, x_pred_tensor.shape[0], bs_ae):
                    chunk = x_pred_tensor[i : i + bs_ae]
                    decoded_chunk = ae_model.decode(chunk).sample
                    decoded_chunks.append(decoded_chunk)
                x_pred_tensor = torch.cat(decoded_chunks, dim=0)
            else:
                print(f"{debug_print_prefix}Warning: AE model not available for decoding in partial eval.")
                x_pred_tensor = torch.rand(B * S * T, 1, H_true, W_true, device=device) * 255.0

            if normalized_autoencoder:
                x_pred_tensor = x_pred_tensor * 255.0

            if torch.isnan(x_pred_tensor).any():
                print(f"{debug_print_prefix} WARNING: NaN values found in x_pred after decode (likely due to FP16) - Please rerun with fp16: false")

            x_pred_tensor = x_pred_tensor.reshape(B, S, T, 1, H_true, W_true)
            x_pred_tensor = x_pred_tensor.permute(0, 1, 2, 4, 5, 3)
            if x_pred_tensor.shape[-1] == 1:
                x_pred_tensor = x_pred_tensor.squeeze(-1)

            x_pred_np = x_pred_tensor.cpu().numpy().astype(np.float32)

            y_pred_batches.append(x_pred_np)
            y_true_batches.append(x_true_np)

            count += B
            if count >= partial_evaluation_batches * val_sample_loader.batch_size:
                break
        eval_bar.close()

        if not y_pred_batches:
            print(f"{debug_print_prefix}No batches processed during deterministic partial evaluation.")
            return None

        y_pred_array = np.concatenate(y_pred_batches, axis=0)
        y_true_array = np.concatenate(y_true_batches, axis=0)
        y_pred_array = post_process_samples(y_pred_array, clamp_min=0.0, clamp_max=255.0)

        for metrics_accumulator in metrics_accumulators:
            metrics_accumulator.update(y_true_array, y_pred_array)

        results = calculate_metrics(num_lead_times=lead_time, metrics_accumulators=metrics_accumulators, thresholds=thresholds)
        EMA_SUFFIX = "(EMA)" if ema_model_evaluated else ""
        print(
            f"{debug_print_prefix}Partial Results (deterministic) {EMA_SUFFIX}: "
            f"MSE: {results.get('mse_from_mean_mean', 'N/A')}, CSI-M: {results.get('csi_from_mean_m', 'N/A')}, "
            f"HSS-M: {results.get('hss_from_mean_m', 'N/A')}, FAR-M: {results.get('far_from_mean_m', 'N/A')}"
        )

        if enable_wandb and wandb_instance:
            ema_suffix_wandb = "_EMA" if ema_model_evaluated else ""
            wandb_instance.log(
                {
                    f"partial_mse{ema_suffix_wandb}": results["mse_from_mean_mean"],
                    f"partial_csi_m{ema_suffix_wandb}": results["csi_from_mean_m"],
                    f"partial_hss_m{ema_suffix_wandb}": results["hss_from_mean_m"],
                    f"partial_far_m{ema_suffix_wandb}": results["far_from_mean_m"],
                },
                step=global_step,
            )

    return results


def run_training(
    config,
    train_dataset,
    val_dataset,
    device,
    run_id=None,
    ae_model=None,
    val_sample_loader=None,
    enable_wandb=None,
):
    """Runs the full single-GPU CFM training loop.

    Parameterized by already-constructed datasets (rather than building them from
    file paths internally) so it can be driven both by `main()` (real SEVIR data)
    and by tests (synthetic in-memory datasets), without duplicating the loop.

    Returns
    -------
    dict with the final `global_step` and the early-stopping `best_metric`.
    """
    DEBUG_MODE = config.run_params.debug_mode
    BACKBONE_TYPE = config.run_params.backbone_type
    RUN_STRING = config.run_params.run_string
    TRAINING_MODE = config.run_params.get("training_mode", "cfm")
    if TRAINING_MODE not in ("cfm", "deterministic"):
        raise ValueError(f"Unknown training_mode: {TRAINING_MODE!r}. Expected 'cfm' or 'deterministic'.")
    if enable_wandb is None:
        enable_wandb = config.run_params.enable_wandb

    if TRAINING_MODE == "deterministic":
        probabilistic_samples = config.get("test_params", {}).get("probabilistic_samples", 1)
        if probabilistic_samples and probabilistic_samples > 1:
            warnings.warn(
                f"training_mode='deterministic' but test_params.probabilistic_samples="
                f"{probabilistic_samples} (>1). Deterministic mode produces exactly one "
                "prediction per input; computing probabilistic ensemble metrics (e.g. CRPS) "
                "over duplicated copies of that same prediction would be misleading. Partial "
                "evaluation below always uses a single sample regardless of this config value; "
                "set test_params.probabilistic_samples=1 for the final evaluation too."
            )

    PARTIAL_EVALUATION = config.partial_evaluation_params.partial_evaluation and val_sample_loader is not None
    PARTIAL_EVALUATION_INTERVAL = config.partial_evaluation_params.partial_evaluation_interval
    PARTIAL_EVALUATION_BATCHES = config.partial_evaluation_params.partial_evaluation_batches
    CARTOPY_FEATURES = config.partial_evaluation_params.cartopy_features
    NORMALIZED_AUTOENCODER = config.autoencoder_params.normalized_autoencoder

    BATCH_SIZE = config.training_params.micro_batch_size
    LEARNING_RATE = config.optimizer_params.learning_rate
    NUM_EPOCHS = config.training_params.num_epochs
    NUM_WORKERS = config.training_params.num_workers
    EARLY_STOPPING_PATIENCE = config.training_params.early_stopping_patience
    EARLY_STOPPING_METRIC = config.training_params.early_stopping_metric
    GRAD_CLIP = config.training_params.gradient_clip_val
    GRAD_ACCUMULATION_STEPS = config.training_params.grad_accumulation_steps
    USE_FP16 = config.training_params.fp16
    FLOW_MATCHING_METHOD = config.flow_matching_params.flow_matching_method

    if EARLY_STOPPING_METRIC in ["partial_csi_m", "partial_mse"] and not PARTIAL_EVALUATION:
        raise ValueError(
            f"Early stopping metric {EARLY_STOPPING_METRIC} requires partial evaluation to be enabled"
        )

    OPTIMIZER_TYPE = config.optimizer_params.optimizer_type
    WEIGHT_DECAY = config.optimizer_params.weight_decay

    SCHEDULER_TYPE = config.scheduler_params.scheduler_type
    LR_PLATEAU_FACTOR = config.scheduler_params.lr_plateau_factor
    LR_PLATEAU_PATIENCE = config.scheduler_params.lr_plateau_patience
    LR_COSINE_WARMUP_ITER_PERCENTAGE = config.scheduler_params.lr_cosine_warmup_iter_percentage
    LR_COSINE_MIN_WARMUP_LR_RATIO = config.scheduler_params.lr_cosine_min_warmup_lr_ratio
    LR_COSINE_MIN_LR_RATIO = config.scheduler_params.lr_cosine_min_lr_ratio

    EMA_MODEL_SAVING = config.ema_model_saving_params.ema_model_saving
    EMA_MODEL_SAVING_DECAY = config.ema_model_saving_params.ema_model_saving_decay

    if run_id is None:
        run_id = (
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + RUN_STRING
            + "_"
            + namegenerator.gen()
        )

    if enable_wandb:
        wandb.init(project="sevir-nowcasting-cfm-single-gpu", name=run_id, config=OmegaConf.to_container(config))

    artifacts_folder = f"artifacts/sevir/smaat_cfm/{run_id}"
    metrics_folder = f"{artifacts_folder}/plots/metrics"
    model_save_dir = f"{artifacts_folder}/models"
    model_save_path = os.path.join(model_save_dir, "early_stopping_model.pt")
    os.makedirs(metrics_folder, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Warning: CPU is used for computation!")

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dynamic_encoded_sequential_collate,
        num_workers=NUM_WORKERS if not DEBUG_MODE else 0,
        pin_memory=True if not DEBUG_MODE else False,
        drop_last=True,
        persistent_workers=True if (not DEBUG_MODE and NUM_WORKERS > 0) else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dynamic_encoded_sequential_collate,
        num_workers=NUM_WORKERS if not DEBUG_MODE else 0,
        pin_memory=True if not DEBUG_MODE else False,
        drop_last=False,
        persistent_workers=True if (not DEBUG_MODE and NUM_WORKERS > 0) else False,
    )

    # Input/output dims are derived from the actual first batch shape, not hardcoded.
    shape_probe_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dynamic_encoded_sequential_collate, num_workers=0
    )
    input_shape = output_shape = None
    for batch in shape_probe_loader:
        inputs_cpu, outputs_cpu, _ = batch
        input_shape = inputs_cpu.shape
        output_shape = outputs_cpu.shape
        break
    del shape_probe_loader
    if input_shape is None or output_shape is None:
        raise RuntimeError("Could not determine input/output shapes from the dataset.")

    mean_std_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE * 4,
        shuffle=False,
        collate_fn=dynamic_encoded_sequential_collate,
        num_workers=0,
        pin_memory=False,
    )
    mean, std = compute_mean_std(mean_std_loader, channel_last=True)
    del mean_std_loader
    print(f"Computed Mean: {mean}, Std: {std}")

    input_shape_flowcast = (input_shape[1], input_shape[2], input_shape[3], input_shape[4])
    output_shape_flowcast = (output_shape[1], output_shape[2], output_shape[3], output_shape[4])

    model = build_backbone(BACKBONE_TYPE, input_shape_flowcast, output_shape_flowcast, config, mean, std)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {BACKBONE_TYPE}, parameter count: {n_params} ({n_params / 1e6:.2f}M)")
    if enable_wandb:
        wandb.config.update({"backbone_type": BACKBONE_TYPE, "backbone_param_count": n_params})

    model = model.to(device)

    ema_model = None
    if EMA_MODEL_SAVING:
        ema_model = copy.deepcopy(model).to(device)

    num_batches_per_epoch = len(train_loader)
    total_num_steps = int(NUM_EPOCHS * num_batches_per_epoch)

    if OPTIMIZER_TYPE == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_TYPE == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Invalid optimizer type: {OPTIMIZER_TYPE}")

    warmup_iter = max(int(np.round(LR_COSINE_WARMUP_ITER_PERCENTAGE * total_num_steps)), 1)

    if SCHEDULER_TYPE == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_PLATEAU_FACTOR, patience=LR_PLATEAU_PATIENCE
        )
    elif SCHEDULER_TYPE == "cosine":
        warmup_scheduler = LambdaLR(
            optimizer, lr_lambda=warmup_lambda(warmup_steps=warmup_iter, min_lr_ratio=LR_COSINE_MIN_WARMUP_LR_RATIO)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(total_num_steps - warmup_iter, 1), eta_min=LR_COSINE_MIN_LR_RATIO * LEARNING_RATE
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iter])
    else:
        raise ValueError(f"Invalid scheduler type: {SCHEDULER_TYPE}")

    criterion = nn.MSELoss(reduction="none")

    sigma = config.flow_matching_params.sigma
    if FLOW_MATCHING_METHOD != "vanilla":
        raise ValueError(f"Invalid flow matching method: {FLOW_MATCHING_METHOD}")
    flow_matcher = ConditionalFlowMatcher(sigma=sigma)

    if EARLY_STOPPING_METRIC == "val_loss":
        metric_direction = "minimize"
        best_metric_init = float("inf")
    elif EARLY_STOPPING_METRIC == "partial_csi_m":
        metric_direction = "maximize"
        best_metric_init = -np.inf
    elif EARLY_STOPPING_METRIC == "partial_mse":
        metric_direction = "minimize"
        best_metric_init = float("inf")
    else:
        metric_direction = "minimize"
        best_metric_init = float("inf")

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        verbose=True,
        path=model_save_path,
        initial_best_metric=best_metric_init,
        metric_direction=metric_direction,
    )

    global_step = 0
    scaler = torch.amp.GradScaler(device=device.type, enabled=USE_FP16)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_accum = 0.0
        train_count = 0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(train_bar):
            inputs, outputs, metadata = batch
            x1 = outputs.to(device, non_blocking=True)
            x0_cond = inputs.to(device, non_blocking=True)

            x0_cond = model.normalize(x0_cond)
            x1 = model.normalize(x1)

            with torch.amp.autocast(device_type=device.type, enabled=USE_FP16):
                pred, target = compute_model_output_and_target(model, flow_matcher, x0_cond, x1, TRAINING_MODE, device)
                raw_per_sample_loss = criterion(pred, target)
                dims_to_reduce = list(range(1, raw_per_sample_loss.ndim))
                final_batch_loss = raw_per_sample_loss.mean(dim=dims_to_reduce).mean()

            scaled_loss = final_batch_loss / GRAD_ACCUMULATION_STEPS
            scaler.scale(scaled_loss).backward()

            raw_loss_value = final_batch_loss.item()
            train_loss_accum += raw_loss_value
            train_count += 1

            if (batch_idx + 1) % GRAD_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if SCHEDULER_TYPE == "cosine":
                    scheduler.step()

                if EMA_MODEL_SAVING and ema_model is not None:
                    ema(model, ema_model, EMA_MODEL_SAVING_DECAY)

                if enable_wandb:
                    wandb.log(
                        {"training_loss_step": raw_loss_value, "learning_rate": optimizer.param_groups[0]["lr"]},
                        step=global_step,
                    )

            global_step += BATCH_SIZE
            train_bar.set_postfix({"training_loss": f"{raw_loss_value:.4f}"})

            if DEBUG_MODE and batch_idx >= 2:
                break
        train_bar.close()

        avg_train_loss = train_loss_accum / train_count if train_count > 0 else 0.0

        model.eval()
        if ema_model is not None:
            ema_model.eval()

        val_loss_accum = 0.0
        val_count = 0
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                inputs, outputs, metadata = batch
                x1 = outputs.to(device, non_blocking=True)
                x0_cond = inputs.to(device, non_blocking=True)

                x0_cond = model.normalize(x0_cond)
                x1 = model.normalize(x1)

                with torch.amp.autocast(device_type=device.type, enabled=USE_FP16):
                    pred, target = compute_model_output_and_target(model, flow_matcher, x0_cond, x1, TRAINING_MODE, device)
                    loss = criterion(pred, target).mean()

                val_loss_accum += loss.item() * inputs.size(0)
                val_count += inputs.size(0)
                val_bar.set_postfix({"validation_loss": f"{loss.item():.4f}"})

                if DEBUG_MODE and val_count // BATCH_SIZE >= 3:
                    break
        val_bar.close()
        avg_val_loss = val_loss_accum / val_count if val_count > 0 else 0.0

        if SCHEDULER_TYPE == "plateau":
            scheduler.step(avg_val_loss)

        partial_eval_results = None
        if PARTIAL_EVALUATION and (epoch % PARTIAL_EVALUATION_INTERVAL == 0):
            eval_model = ema_model if (EMA_MODEL_SAVING and ema_model is not None) else model
            eval_model.eval()
            if TRAINING_MODE == "cfm":
                partial_eval_results = partial_evaluate_model(
                    model=eval_model,
                    device=device,
                    val_sample_loader=val_sample_loader,
                    thresholds=THRESHOLDS,
                    global_step=global_step,
                    epoch=epoch,
                    ae_model=ae_model,
                    normalized_autoencoder=NORMALIZED_AUTOENCODER,
                    use_fp16=USE_FP16,
                    partial_evaluation_batches=PARTIAL_EVALUATION_BATCHES,
                    lead_time=config.data_params.lead_time,
                    enable_wandb=enable_wandb,
                    wandb_instance=wandb if enable_wandb else None,
                    debug_print_prefix="[single-gpu] ",
                    plots_folder=metrics_folder,
                    cartopy_features=CARTOPY_FEATURES,
                    ema_model_evaluated=EMA_MODEL_SAVING and ema_model is not None,
                    batch_size_autoencoder=None if BATCH_SIZE > 2 else BATCH_SIZE,
                )
            else:
                partial_eval_results = partial_evaluate_model_deterministic(
                    model=eval_model,
                    device=device,
                    val_sample_loader=val_sample_loader,
                    thresholds=THRESHOLDS,
                    global_step=global_step,
                    epoch=epoch,
                    ae_model=ae_model,
                    normalized_autoencoder=NORMALIZED_AUTOENCODER,
                    use_fp16=USE_FP16,
                    partial_evaluation_batches=PARTIAL_EVALUATION_BATCHES,
                    lead_time=config.data_params.lead_time,
                    enable_wandb=enable_wandb,
                    wandb_instance=wandb if enable_wandb else None,
                    debug_print_prefix="[single-gpu] ",
                    ema_model_evaluated=EMA_MODEL_SAVING and ema_model is not None,
                    batch_size_autoencoder=None if BATCH_SIZE > 2 else BATCH_SIZE,
                )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Finished Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        if enable_wandb:
            wandb.log(
                {"epoch": epoch, "avg_training_loss": avg_train_loss, "avg_validation_loss": avg_val_loss, "learning_rate": current_lr},
                step=global_step,
            )

        if EARLY_STOPPING_METRIC == "val_loss":
            current_metric = avg_val_loss
        elif EARLY_STOPPING_METRIC == "partial_mse":
            current_metric = partial_eval_results.get("mse_from_mean_mean", float("inf")) if partial_eval_results else float("inf")
        elif EARLY_STOPPING_METRIC == "partial_csi_m":
            current_metric = partial_eval_results.get("csi_from_mean_m", -np.inf) if partial_eval_results else -np.inf
        else:
            current_metric = avg_val_loss

        model_to_save = ema_model if (EMA_MODEL_SAVING and ema_model is not None) else model
        early_stopping(current_metric, model_to_save, optimizer, epoch, global_step)

        if early_stopping.early_stop:
            print("Early stopping condition met. Finalizing training.")
            break

    if enable_wandb:
        wandb.finish()

    return {"global_step": global_step, "best_metric": early_stopping.best_metric}


def main():
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DEBUG_MODE = config.run_params.debug_mode
    PARTIAL_EVALUATION = config.partial_evaluation_params.partial_evaluation
    AUTOENCODER_CHECKPOINT = config.autoencoder_params.autoencoder_checkpoint

    if PARTIAL_EVALUATION and AUTOENCODER_CHECKPOINT is None:
        raise ValueError("Partial Evaluation is enabled but Autoencoder Checkpoint is not provided")

    train_dataset = DynamicEncodedSequentialSevirDataset(
        meta_csv=args.train_meta,
        data_file=args.train_file,
        data_type="vil",
        raw_seq_len=49,
        lag_time=config.data_params.lag_time,
        lead_time=config.data_params.lead_time,
        time_spacing=config.data_params.time_spacing,
        stride=12,
        channel_last=True,
        debug_mode=DEBUG_MODE,
        transform=None,
    )
    val_dataset = DynamicEncodedSequentialSevirDataset(
        meta_csv=args.val_meta,
        data_file=args.val_file,
        data_type="vil",
        raw_seq_len=49,
        lag_time=config.data_params.lag_time,
        lead_time=config.data_params.lead_time,
        time_spacing=config.data_params.time_spacing,
        stride=12,
        channel_last=True,
        debug_mode=DEBUG_MODE,
        transform=None,
    )

    ae_model = None
    val_sample_loader = None
    if PARTIAL_EVALUATION:
        if not os.path.exists(AUTOENCODER_CHECKPOINT):
            raise FileNotFoundError(f"AE model not found at {AUTOENCODER_CHECKPOINT}")

        from diffusers.models.autoencoders import AutoencoderKL

        ae_model = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            down_block_types=config.autoencoder_params.down_block_types,
            up_block_types=config.autoencoder_params.up_block_types,
            block_out_channels=config.autoencoder_params.block_out_channels,
            act_fn=config.autoencoder_params.act_fn,
            latent_channels=config.autoencoder_params.latent_channels,
            norm_num_groups=config.autoencoder_params.norm_num_groups,
            layers_per_block=config.autoencoder_params.layers_per_block,
        )
        checkpoint = torch.load(AUTOENCODER_CHECKPOINT, map_location=device)
        new_state_dict = {
            (k.replace("module.", "") if k.startswith("module.") else k): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        ae_model.load_state_dict(new_state_dict)
        ae_model = ae_model.to(device)
        ae_model.eval()

        val_sample_dataset = DynamicSequentialSevirDataset(
            meta_csv=args.partial_evaluation_meta,
            data_file=args.partial_evaluation_file,
            data_type="vil",
            raw_seq_len=49,
            lag_time=config.data_params.lag_time,
            lead_time=config.data_params.lead_time,
            time_spacing=config.data_params.time_spacing,
            stride=12,
            channel_last=False,
            debug_mode=DEBUG_MODE,
        )
        val_sample_loader = DataLoader(
            val_sample_dataset,
            batch_size=config.training_params.micro_batch_size // 4 or 1,
            shuffle=False,
            collate_fn=dynamic_sequential_collate,
            num_workers=config.training_params.num_workers if not DEBUG_MODE else 0,
            pin_memory=True if not DEBUG_MODE else False,
        )

    run_training(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        ae_model=ae_model,
        val_sample_loader=val_sample_loader,
    )


if __name__ == "__main__":
    main()
