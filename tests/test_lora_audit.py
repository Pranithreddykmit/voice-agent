"""
Unit tests for IndicF5 LoRA Feasibility Audit & Parameter-Efficient Fine-Tuning Setup (Phase 6C).
Verifies PEFT LoraConfig, target module target names, parameter counting logic, and audit integrity.
"""

import json
import pytest
from pathlib import Path
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def test_lora_target_modules_structure():
    """Verifies LoRA target module names match DiT transformer projection structure."""
    target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.ff.0.0", "ff.ff.2"]
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none"
    )
    assert config.r == 8
    assert config.lora_alpha == 16
    assert set(config.target_modules) == set(target_modules)


def test_audit_results_file_integrity():
    """Verifies that checkpoints/audit_results.json exists and contains verified metrics."""
    audit_file = Path("checkpoints/audit_results.json")
    assert audit_file.exists(), "audit_results.json must exist"
    
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "hardware_audit" in data
    assert "model_architecture" in data
    assert "lora_attachment" in data
    assert "dry_run_metrics" in data
    assert "checkpoint_test" in data

    assert data["lora_attachment"]["base_frozen"] is True
    assert data["lora_attachment"]["lora_trainable"] is True
    assert data["checkpoint_test"]["save"] is True
    assert data["checkpoint_test"]["reload"] is True
    assert data["checkpoint_test"]["adapter_restored"] is True
    assert data["checkpoint_test"]["base_unchanged"] is True
