"""
IndicF5 Telugu Speech LoRA Fine-Tuning Pipeline (Phase 6C).
Executes controlled parameter-efficient adaptation on the hybrid Telugu dataset (IndicTTS + SYSPIN)
using Diffusion Transformer (DiT) flow matching on Apple Silicon MPS hardware.
"""

import os
import sys
import math
import json
import time
import shutil
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import soundfile as sf
import numpy as np

# Ensure ai-service is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
AI_SERVICE_DIR = BASE_DIR / "ai-service"
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from f5_tts.model import CFM, DiT
from f5_tts.model.dataset import CustomDataset, collate_fn
import peft
from peft import LoraConfig, get_peft_model, PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("training.indicf5_lora")

DEFAULT_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "ff.ff.0.0", "ff.ff.2"]


def set_seed(seed: int = 42):
    """Sets deterministic random seeds across all libraries."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_ram_mb() -> float:
    """Returns current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def detect_optimal_device() -> torch.device:
    """Detects available acceleration hardware (MPS / CUDA / CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def compute_base_model_hash(model: nn.Module) -> str:
    """Computes a SHA256 checksum across all non-LoRA base parameter weights."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if "lora" not in name:
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def load_indicf5_base_model(
    repo_id: str = "ai4bharat/IndicF5"
) -> Tuple[DiT, Dict[str, Any], Dict[str, int], int, Dict[str, Any]]:
    """Loads frozen ai4bharat/IndicF5 checkpoint and initializes DiT backbone."""
    logger.info(f"Loading base weights from {repo_id}...")
    vocab_path = hf_hub_download(repo_id=repo_id, filename="checkpoints/vocab.txt")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_char_map = {ch.strip(): i for i, ch in enumerate(f)}

    raw_sd = load_file(ckpt_path)
    cleaned_sd = {}
    for k, v in raw_sd.items():
        clean_k = k.replace("ema_model._orig_mod.", "").replace("ema_model.", "").replace("_orig_mod.", "")
        cleaned_sd[clean_k] = v

    weight_shape = cleaned_sd["transformer.text_embed.text_embed.weight"].shape[0]

    model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
        mel_dim=100, text_num_embeds=weight_shape - 1
    )

    transformer = DiT(**model_cfg)
    transformer.load_state_dict({k.replace("transformer.", ""): v for k, v in cleaned_sd.items() if k.startswith("transformer.")}, strict=False)

    return transformer, cleaned_sd, vocab_char_map, weight_shape, model_cfg


def apply_lora_to_dit(
    transformer: DiT,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Wraps DiT backbone with PEFT LoRA adapter and freezes base parameters."""
    targets = target_modules or DEFAULT_TARGET_MODULES

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=targets,
        lora_dropout=lora_dropout,
        bias="none"
    )

    lora_dit = get_peft_model(transformer, lora_config)

    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in lora_dit.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in lora_dit.parameters() if not p.requires_grad)

    param_stats = {
        "total_dit_params": total_params,
        "trainable_lora_params": trainable_params,
        "frozen_base_params": frozen_params,
        "trainable_percentage": round((trainable_params / total_params) * 100.0, 4),
        "target_modules": targets,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout
    }

    logger.info(f"LoRA Adapter Attached: {trainable_params:,} trainable params ({param_stats['trainable_percentage']}%)")
    return lora_dit, param_stats


def create_dataset_loader(
    csv_path: str,
    batch_size: int = 2,
    shuffle: bool = True
) -> Tuple[DataLoader, int]:
    """Creates a PyTorch DataLoader from an IndicF5 pipe-delimited metadata CSV."""
    import csv as pycsv

    records = []
    durations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = pycsv.DictReader(f, delimiter="|")
        for row in reader:
            dur = sf.info(row["audio_file"]).duration
            records.append({
                "audio_path": row["audio_file"],
                "text": row["text"],
                "duration": dur
            })
            durations.append(dur)

    dataset = CustomDataset(
        custom_dataset=records,
        durations=durations,
        target_sample_rate=24000,
        hop_length=256,
        n_mel_channels=100
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=False
    )
    return loader, len(dataset)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.05
):
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_validation_loss(cfm: CFM, val_loader: DataLoader, device: torch.device) -> float:
    """Computes CFM Flow-Matching validation loss across the validation split."""
    cfm.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].permute(0, 2, 1).to(device)
            lens = batch["mel_lengths"].to(device)
            text = batch["text"]
            loss, cond, pred = cfm(mel, text=text, lens=lens)
            val_losses.append(float(loss.item()))
    cfm.train()
    return float(np.mean(val_losses)) if val_losses else 0.0


def train_lora(
    train_csv: str = "dataset/telugu_tts/processed_hybrid/train.csv",
    val_csv: str = "dataset/telugu_tts/processed_hybrid/val.csv",
    output_dir: str = "models/indicf5_telugu_lora",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lr: float = 1e-5,
    batch_size: int = 2,
    grad_accum_steps: int = 2,
    epochs: int = 50,
    warmup_steps: int = 100,
    seed: int = 42,
    device_override: Optional[str] = None
) -> Dict[str, Any]:
    """Executes the complete controlled LoRA fine-tuning run for IndicF5."""
    set_seed(seed)
    device = torch.device(device_override) if device_override else detect_optimal_device()
    logger.info(f"Execution hardware device: {device}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ram_start = get_ram_mb()
    logger.info(f"Initial Process RAM: {ram_start:.2f} MB")

    # 1. Load Base Model
    base_dit, cleaned_sd, vocab_char_map, weight_shape, model_cfg = load_indicf5_base_model()
    
    # Check base model fingerprint before training
    pre_train_base_hash = compute_base_model_hash(base_dit)
    logger.info(f"Base weights pre-training SHA256: {pre_train_base_hash[:16]}...")

    # 2. Attach PEFT LoRA
    lora_dit, param_stats = apply_lora_to_dit(
        transformer=base_dit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES
    )

    # 3. Build CFM Wrapper
    cfm = CFM(
        transformer=lora_dit,
        mel_spec_kwargs=dict(
            target_sample_rate=24000, n_mel_channels=100, hop_length=256,
            win_length=1024, n_fft=1024, mel_spec_type="vocos"
        ),
        vocab_char_map=vocab_char_map
    )
    cfm.load_state_dict(cleaned_sd, strict=False)
    cfm.to(device)

    # 4. Load Data
    train_loader, n_train = create_dataset_loader(train_csv, batch_size=batch_size, shuffle=True)
    val_loader, n_val = create_dataset_loader(val_csv, batch_size=batch_size, shuffle=False)
    logger.info(f"Dataset Loaded: Train ({n_train} samples), Validation ({n_val} samples)")

    # Calculate total training steps
    steps_per_epoch = len(train_loader)
    total_training_steps = epochs * steps_per_epoch
    effective_opt_steps = total_training_steps // grad_accum_steps
    logger.info(f"Total training batches: {total_training_steps} across {epochs} epochs ({effective_opt_steps} optimizer updates)")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in lora_dit.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # Initial Validation Loss
    initial_val_loss = evaluate_validation_loss(cfm, val_loader, device)
    logger.info(f"Pre-Training Initial Validation Loss: {initial_val_loss:.6f}")

    # 6. Training Loop
    cfm.train()
    global_step = 0
    opt_step = 0
    best_val_loss = float("inf")
    best_epoch = -1
    peak_ram = ram_start

    epoch_metrics: List[Dict[str, Any]] = []
    step_losses: List[float] = []

    t_train_start = time.perf_counter()
    step_times: List[float] = []

    logger.info(f"=== Starting Controlled LoRA Training ({epochs} Epochs) ===")

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()
        epoch_train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            t_s0 = time.perf_counter()
            mel = batch["mel"].permute(0, 2, 1).to(device)
            lens = batch["mel_lengths"].to(device)
            text = batch["text"]

            # Forward pass (Flow-Matching MSE)
            loss, cond, pred = cfm(mel, text=text, lens=lens)
            loss_val = float(loss.item())
            epoch_train_losses.append(loss_val)
            step_losses.append(loss_val)

            # Backward pass with gradient accumulation
            (loss / grad_accum_steps).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(lora_dit.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

            if device.type == "mps":
                torch.mps.synchronize()

            step_time = time.perf_counter() - t_s0
            step_times.append(step_time)
            global_step += 1

            cur_ram = get_ram_mb()
            if cur_ram > peak_ram:
                peak_ram = cur_ram

        # Epoch Validation & Loss Aggregation
        train_loss_avg = float(np.mean(epoch_train_losses))
        val_loss_avg = evaluate_validation_loss(cfm, val_loader, device)
        cur_lr = scheduler.get_last_lr()[0]
        epoch_elapsed = time.perf_counter() - t_epoch_start

        # Periodic cache cleanup to prevent memory accumulation
        import gc
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        # Track Best Validation Checkpoint
        is_best = val_loss_avg < best_val_loss
        if is_best:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            # Save best adapter
            best_adapter_dir = out_path / "best_adapter"
            best_adapter_dir.mkdir(parents=True, exist_ok=True)
            lora_dit.save_pretrained(str(best_adapter_dir))

        # Check for potential overfitting indicator
        overfitting_flag = False
        if epoch > 10 and (val_loss_avg - best_val_loss) > 0.15:
            overfitting_flag = True

        epoch_rec = {
            "epoch": epoch,
            "global_step": global_step,
            "opt_step": opt_step,
            "train_loss": round(train_loss_avg, 6),
            "val_loss": round(val_loss_avg, 6),
            "lr": round(cur_lr, 8),
            "epoch_sec": round(epoch_elapsed, 2),
            "is_best": is_best,
            "overfitting_flag": overfitting_flag
        }
        epoch_metrics.append(epoch_rec)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or is_best:
            best_mark = " ★ BEST" if is_best else ""
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss_avg:.6f} | "
                f"Val Loss: {val_loss_avg:.6f} | LR: {cur_lr:.2e} | "
                f"Time: {epoch_elapsed:.1f}s{best_mark}"
            )

    total_train_sec = time.perf_counter() - t_train_start
    avg_step_sec = float(np.mean(step_times)) if step_times else 0.0

    logger.info(f"=== Training Completed in {total_train_sec:.2f}s ({total_train_sec/60:.2f}m) ===")
    logger.info(f"Best Validation Loss: {best_val_loss:.6f} at Epoch {best_epoch}")

    # 7. Final Model Checkpoint Save
    final_adapter_dir = out_path / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    lora_dit.save_pretrained(str(final_adapter_dir))

    # Also copy best adapter directly to root of output_dir
    best_adapter_src = out_path / "best_adapter"
    for f_path in best_adapter_src.glob("*"):
        shutil.copy(str(f_path), str(out_path / f_path.name))

    # 8. Post-Training Base Weight Integrity Verification
    post_train_base_hash = compute_base_model_hash(lora_dit)
    base_weights_modified = (pre_train_base_hash != post_train_base_hash)
    logger.info(f"Base weights post-training SHA256: {post_train_base_hash[:16]}...")
    logger.info(f"Base weights modified: {'YES (ERROR)' if base_weights_modified else 'NO (VERIFIED UNCHANGED)'}")

    # 9. Adapter Reload Verification
    test_dit = DiT(**model_cfg)
    test_dit.load_state_dict({k.replace("transformer.", ""): v for k, v in cleaned_sd.items() if k.startswith("transformer.")}, strict=False)
    reloaded_peft = PeftModel.from_pretrained(test_dit, str(out_path))

    reloaded_lora_params = sum(p.numel() for n, p in reloaded_peft.named_parameters() if "lora_" in n)
    reloaded_base_frozen = all(not p.requires_grad for n, p in reloaded_peft.named_parameters() if "lora_" not in n)
    adapter_reload_pass = (reloaded_peft is not None) and (reloaded_lora_params == param_stats["trainable_lora_params"])

    logger.info(f"Adapter Reload Verification: {'PASS' if adapter_reload_pass else 'FAIL'}")
    logger.info(f"Reloaded LoRA Parameters: {reloaded_lora_params:,} (Frozen Base: {reloaded_base_frozen})")

    # 10. Checkpoint Size & Files
    saved_files = [f.name for f in out_path.glob("*") if f.is_file()]
    adapter_size_mb = sum((out_path / fname).stat().st_size for fname in saved_files) / (1024 * 1024)

    # 11. Save Training Metrics & Config
    training_config = {
        "model_name": "ai4bharat/IndicF5",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": DEFAULT_TARGET_MODULES,
        "learning_rate": lr,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "device": str(device),
        "precision": "float32 (MPS)",
        "train_csv": train_csv,
        "val_csv": val_csv,
        "n_train_samples": n_train,
        "n_val_samples": n_val
    }

    with open(out_path / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    summary_metrics = {
        "status": "training_completed",
        "epochs_completed": epochs,
        "total_batches": global_step,
        "optimizer_steps": opt_step,
        "initial_train_loss": round(epoch_metrics[0]["train_loss"], 6) if epoch_metrics else 0.0,
        "final_train_loss": round(epoch_metrics[-1]["train_loss"], 6) if epoch_metrics else 0.0,
        "initial_val_loss": round(initial_val_loss, 6),
        "final_val_loss": round(epoch_metrics[-1]["val_loss"], 6) if epoch_metrics else 0.0,
        "best_val_loss": round(best_val_loss, 6),
        "best_epoch": best_epoch,
        "total_training_time_sec": round(total_train_sec, 2),
        "average_step_time_sec": round(avg_step_sec, 4),
        "peak_ram_mb": round(peak_ram, 2),
        "adapter_size_mb": round(adapter_size_mb, 2),
        "param_stats": param_stats,
        "base_weights_modified": base_weights_modified,
        "adapter_reload_pass": adapter_reload_pass,
        "saved_files": saved_files,
        "epoch_metrics": epoch_metrics
    }

    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(f"Metrics written to {out_path / 'metrics.json'}")
    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IndicF5 Telugu LoRA Adapter")
    parser.add_argument("--train-csv", type=str, default="dataset/telugu_tts/processed_hybrid/train.csv")
    parser.add_argument("--val-csv", type=str, default="dataset/telugu_tts/processed_hybrid/val.csv")
    parser.add_argument("--output-dir", type=str, default="models/indicf5_telugu_lora")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_lora(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        lora_r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        epochs=args.epochs,
        warmup_steps=args.warmup,
        seed=args.seed
    )
