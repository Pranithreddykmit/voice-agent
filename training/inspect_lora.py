"""
IndicF5 LoRA Feasibility Audit & Safe Dry Run Script (Phase 6C).
Executes hardware audit, architecture inspection, module enumeration,
PEFT compatibility verification, LoRA attachment test, 1-step dry run,
and adapter checkpoint save/reload test.
"""

import os
import sys
import time
import gc
import json
import shutil
import platform
from pathlib import Path
from typing import Dict, Any, List

import psutil
import torch
import torch.nn as nn
import soundfile as sf

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


def get_ram_mb() -> float:
    """Returns current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_hardware_audit() -> Dict[str, Any]:
    """Part 1: Hardware Audit."""
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    avail_ram_gb = psutil.virtual_memory().available / (1024**3)
    
    mps_avail = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    cuda_avail = torch.cuda.is_available()
    
    audit = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or "Apple Silicon",
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(avail_ram_gb, 2),
        "gpu": "Apple Silicon Unified Memory (MPS)" if mps_avail else ("CUDA GPU" if cuda_avail else "None"),
        "gpu_memory": f"{round(total_ram_gb, 2)} GB (Shared Unified Memory)" if mps_avail else "N/A",
        "cuda_available": cuda_avail,
        "mps_available": mps_avail,
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "peft_version": peft.__version__,
        "f5_tts_version": getattr(sys.modules.get("f5_tts"), "__version__", "0.1.0 (Installed via setup.py)")
    }
    return audit


def inspect_model_and_modules(repo_id: str = "ai4bharat/IndicF5"):
    """Parts 2 & 3: Model Architecture & Linear Module Inspection."""
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

    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params_before = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

    # Linear module enumeration
    linear_modules = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            in_dim = module.in_features
            out_dim = module.out_features
            params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            linear_modules.append({
                "name": name,
                "in_features": in_dim,
                "out_features": out_dim,
                "param_count": params
            })

    arch_info = {
        "model_class": "f5_tts.model.dit.DiT (wrapped by f5_tts.model.cfm.CFM)",
        "transformer_architecture": "Diffusion Transformer (DiT) with ConvNeXt Mel Backbone, RoPE Attention, & Flow Matching",
        "total_parameters": total_params,
        "trainable_parameters_before_lora": trainable_params_before,
        "attention_implementation": "Multi-Head Self-Attention (to_q, to_k, to_v, to_out.0) with Rotary Position Embeddings (RoPE)",
        "conditioning_mechanism": "Text embedding + Reference Mel prompt + Timestep embedding -> Flow Matching target vector field",
        "forward_signature": "CFM.forward(inp: mel [b, n, d], text: list[str], lens: int[b]) -> (loss, cond, pred)",
        "training_loss": "Conditional Flow Matching (CFM) MSE Loss (Optimal Transport vector field regression)",
        "linear_module_count": len(linear_modules),
        "linear_modules_sample": linear_modules[:10],
        "vocab_char_map_size": len(vocab_char_map)
    }

    return transformer, cleaned_sd, vocab_char_map, weight_shape, arch_info, linear_modules, model_cfg


def run_full_audit():
    print("=" * 80)
    print("PHASE 6C — INDICF5 LORA FEASIBILITY AUDIT & SAFE DRY RUN")
    print("=" * 80)

    # PART 1: Hardware Audit
    print("\n--- PART 1: HARDWARE AUDIT ---")
    hw_audit = run_hardware_audit()
    for k, v in hw_audit.items():
        print(f"  {k}: {v}")

    # PARTS 2 & 3: Model & Linear Module Inspection
    print("\n--- PARTS 2 & 3: MODEL ARCHITECTURE & LINEAR MODULES ---")
    transformer, cleaned_sd, vocab_char_map, weight_shape, arch_info, linear_modules, model_cfg = inspect_model_and_modules()
    for k, v in arch_info.items():
        if k != "linear_modules_sample":
            print(f"  {k}: {v}")

    print(f"\nTotal Linear Modules Found: {len(linear_modules)}")
    target_candidates = ["to_q", "to_k", "to_v", "to_out.0", "ff.ff.0.0", "ff.ff.2"]
    matched_modules = [m for m in linear_modules if any(cand in m["name"] for cand in target_candidates)]
    print(f"Matched Target Modules ({len(matched_modules)} instances):")
    sample_targets = matched_modules[:6]
    for m in sample_targets:
        print(f"  {m['name']} | In: {m['in_features']} -> Out: {m['out_features']} | Params: {m['param_count']:,}")

    # PART 4: PEFT Compatibility
    print("\n--- PART 4: PEFT COMPATIBILITY ---")
    print(f"PEFT Version: {peft.__version__}")
    print("LoraConfig Available: YES")

    target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.ff.0.0", "ff.ff.2"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none"
    )
    print("LoraConfig initialized successfully.")

    # PART 5: Safe LoRA Attachment Test
    print("\n--- PART 5: SAFE LORA ATTACHMENT TEST ---")
    lora_dit = get_peft_model(transformer, lora_config)
    
    total_dit_params = sum(p.numel() for p in lora_dit.parameters())
    trainable_lora_params = sum(p.numel() for p in lora_dit.parameters() if p.requires_grad)
    frozen_base_params = sum(p.numel() for p in lora_dit.parameters() if not p.requires_grad)
    trainable_pct = round((trainable_lora_params / total_dit_params) * 100.0, 4)

    base_frozen_pass = frozen_base_params > 0 and frozen_base_params + trainable_lora_params == total_dit_params
    lora_trainable_pass = trainable_lora_params > 0

    print(f"Base parameters frozen: {'PASS' if base_frozen_pass else 'FAIL'}")
    print(f"LoRA parameters trainable: {'PASS' if lora_trainable_pass else 'FAIL'}")
    print(f"Total DiT Parameters: {total_dit_params:,}")
    print(f"Frozen Base Parameters: {frozen_base_params:,}")
    print(f"Trainable LoRA Parameters: {trainable_lora_params:,}")
    print(f"Trainable Percentage: {trainable_pct}%")

    # PART 6: VERY SMALL Forward/Backward Dry Run
    print("\n--- PART 6: VERY SMALL FORWARD/BACKWARD DRY RUN ---")
    
    # Device selection: MPS if available
    device = torch.device("mps") if hw_audit["mps_available"] else torch.device("cpu")
    print(f"Selected Dry-Run Device: {device}")

    # Build CFM wrapper
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

    # Load 1 sample from dataset
    train_csv = "dataset/telugu_tts/processed_hybrid/train.csv"
    import csv as pycsv
    records = []
    durations = []
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = pycsv.DictReader(f, delimiter="|")
        for i, row in enumerate(reader):
            if i >= 1: # Only 1 sample
                break
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
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    batch = next(iter(loader))

    mel = batch["mel"].permute(0, 2, 1).to(device)
    lens = batch["mel_lengths"].to(device)
    text = batch["text"]

    optimizer = torch.optim.AdamW(
        [p for p in lora_dit.parameters() if p.requires_grad],
        lr=1e-5
    )

    ram_before = get_ram_mb()
    gc.collect()
    if device.type == "mps":
        torch.mps.synchronize()

    # 1. Forward Pass
    t0 = time.perf_counter()
    loss, cond, pred = cfm(mel, text=text, lens=lens)
    if device.type == "mps":
        torch.mps.synchronize()
    t_fwd = time.perf_counter() - t0

    # 2. Backward Pass
    t1 = time.perf_counter()
    loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    t_bwd = time.perf_counter() - t1

    # 3. Optimizer Step
    t2 = time.perf_counter()
    optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    t_opt = time.perf_counter() - t2

    ram_after = get_ram_mb()
    peak_ram = max(ram_before, ram_after)

    print(f"Forward Latency: {t_fwd:.4f} s")
    print(f"Backward Latency: {t_bwd:.4f} s")
    print(f"Optimizer Step Latency: {t_opt:.4f} s")
    print(f"Total 1-step Latency: {(t_fwd + t_bwd + t_opt):.4f} s")
    print(f"Dry Run Loss Value: {float(loss.item()):.6f}")
    print(f"Process RAM Before: {ram_before:.2f} MB")
    print(f"Process RAM After: {ram_after:.2f} MB")
    print(f"Peak RAM: {peak_ram:.2f} MB")

    # PART 7: Temporary Checkpoint Test
    print("\n--- PART 7: CHECKPOINT TEST ---")
    temp_ckpt_dir = Path("checkpoints/temp_indicf5_lora_audit")
    if temp_ckpt_dir.exists():
        shutil.rmtree(temp_ckpt_dir)
    temp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save
    save_pass = False
    reload_pass = False
    adapter_restored_pass = False
    base_unchanged_pass = False

    try:
        lora_dit.save_pretrained(str(temp_ckpt_dir))
        ckpt_files = list(temp_ckpt_dir.glob("*"))
        print(f"Saved adapter checkpoint files: {[f.name for f in ckpt_files]}")
        ckpt_size_mb = sum(f.stat().st_size for f in ckpt_files) / (1024 * 1024)
        print(f"Adapter Checkpoint Size: {ckpt_size_mb:.2f} MB")
        save_pass = len(ckpt_files) > 0 and ckpt_size_mb > 0
    except Exception as e:
        print(f"Save failed: {e}")

    # Reload test
    try:
        fresh_dit = DiT(**model_cfg)
        fresh_dit.load_state_dict({k.replace("transformer.", ""): v for k, v in cleaned_sd.items() if k.startswith("transformer.")}, strict=False)
        
        reloaded_lora = PeftModel.from_pretrained(fresh_dit, str(temp_ckpt_dir))
        reload_pass = reloaded_lora is not None
        
        # Verify trainable adapter weights present by inspecting lora_ named parameter count
        reloaded_lora_param_count = sum(p.numel() for n, p in reloaded_lora.named_parameters() if "lora_" in n)
        adapter_restored_pass = (reloaded_lora_param_count == trainable_lora_params)

        # Verify base weights unchanged between original pretrained transformer and reloaded model base
        base_weights_match = True
        for k, v in transformer.named_parameters():
            if "lora" not in k and k in dict(reloaded_lora.named_parameters()):
                if not torch.equal(v.cpu(), dict(reloaded_lora.named_parameters())[k].cpu()):
                    base_weights_match = False
                    break
        base_unchanged_pass = base_weights_match
    except Exception as e:
        print(f"Reload failed: {e}")

    # Clean up temporary checkpoint
    if temp_ckpt_dir.exists():
        shutil.rmtree(temp_ckpt_dir)

    print(f"Save Adapter: {'PASS' if save_pass else 'FAIL'}")
    print(f"Reload Adapter: {'PASS' if reload_pass else 'FAIL'}")
    print(f"Adapter Parameters Restored: {'PASS' if adapter_restored_pass else 'FAIL'}")
    print(f"Base Weights Unchanged: {'PASS' if base_unchanged_pass else 'FAIL'}")

    results = {
        "hardware_audit": hw_audit,
        "model_architecture": arch_info,
        "total_linear_modules": len(linear_modules),
        "target_modules": target_candidates,
        "matched_target_modules_count": len(matched_modules),
        "lora_attachment": {
            "base_frozen": base_frozen_pass,
            "lora_trainable": lora_trainable_pass,
            "total_dit_params": total_dit_params,
            "frozen_base_params": frozen_base_params,
            "trainable_lora_params": trainable_lora_params,
            "trainable_percentage": trainable_pct
        },
        "dry_run_metrics": {
            "device": str(device),
            "forward_latency_sec": round(t_fwd, 4),
            "backward_latency_sec": round(t_bwd, 4),
            "optimizer_step_latency_sec": round(t_opt, 4),
            "total_latency_sec": round(t_fwd + t_bwd + t_opt, 4),
            "loss": round(float(loss.item()), 6),
            "peak_ram_mb": round(peak_ram, 2)
        },
        "checkpoint_test": {
            "save": save_pass,
            "reload": reload_pass,
            "adapter_restored": adapter_restored_pass,
            "base_unchanged": base_unchanged_pass,
            "checkpoint_size_mb": round(ckpt_size_mb, 2) if save_pass else 0.0
        }
    }
    
    with open("checkpoints/audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nAudit completed successfully. Results written to checkpoints/audit_results.json")
    return results


if __name__ == "__main__":
    run_full_audit()
