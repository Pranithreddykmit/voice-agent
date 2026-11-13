#!/usr/bin/env python3
"""
Forensic TTS Memory Investigation Script for MPS (Apple Silicon GPU).
Instruments memory (RSS and PyTorch MPS/Wired allocations) at every stage:
"""

import os
import sys
import time
import gc
import psutil
import torch
import numpy as np
import soundfile as sf
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-service")))

def get_mem():
    proc = psutil.Process()
    mem_info = proc.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)
    vm = psutil.virtual_memory()
    sys_used_gb = vm.used / (1024**3)
    sys_avail_gb = vm.available / (1024**3)
    
    mps_alloc_mb = 0.0
    mps_driver_mb = 0.0
    if torch.backends.mps.is_available():
        if hasattr(torch.mps, "current_allocated_memory"):
            try:
                mps_alloc_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            except Exception:
                pass
        if hasattr(torch.mps, "driver_allocated_memory"):
            try:
                mps_driver_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
            except Exception:
                pass
            
    return {
        "rss_mb": rss_mb,
        "vms_mb": vms_mb,
        "mps_alloc_mb": mps_alloc_mb,
        "mps_driver_mb": mps_driver_mb,
        "sys_used_gb": sys_used_gb,
        "sys_avail_gb": sys_avail_gb,
    }

def print_mem_stage(stage_name: str, prev_mem: dict):
    curr = get_mem()
    diff_rss = curr["rss_mb"] - prev_mem["rss_mb"]
    print(f"[{stage_name:<35}] RSS: {curr['rss_mb']:>8.2f} MB (delta: {diff_rss:>+7.2f} MB) | MPS Alloc: {curr['mps_alloc_mb']:>7.2f} MB | MPS Driver: {curr['mps_driver_mb']:>7.2f} MB | Sys Used: {curr['sys_used_gb']:.2f} GB")
    return curr

def main():
    print("=" * 80)
    print("  FORENSIC TTS MEMORY INVESTIGATION ON APPLE SILICON M4 (MPS)")
    print("=" * 80)
    
    m0 = get_mem()
    print(f"Initial State -> Process RSS: {m0['rss_mb']:.2f} MB | System Used: {m0['sys_used_gb']:.2f} GB | System Avail: {m0['sys_avail_gb']:.2f} GB\n")
    
    last_m = print_mem_stage("1. Before Model Load", m0)
    
    from app.speech.tts import IndicF5TTSProvider
    tts = IndicF5TTSProvider(adapter_path="models/indicf5_telugu_lora", device="mps")
    last_m = print_mem_stage("2. After IndicF5 Load (MPS)", last_m)
    
    test_text = "నమస్కారం"
    from app.speech.audio import prepare_speech_text
    clean_text = prepare_speech_text(test_text)
    last_m = print_mem_stage("3. After Text Preprocessing", last_m)
    
    last_m = print_mem_stage("4. Before Synthesize Call", last_m)
    
    t0 = time.perf_counter()
    wav_bytes, lat = tts.synthesize(clean_text)
    synth_time = time.perf_counter() - t0
    
    last_m = print_mem_stage("5. After Synthesize Call", last_m)
    
    data, out_sr = sf.read(io.BytesIO(wav_bytes))
    rms = float(np.sqrt(np.mean(data**2)))
    print(f"\nResult: Duration={len(data)/out_sr:.2f}s @ {out_sr}Hz | RMS={rms:.4f} | Latency={lat:.2f}ms ({synth_time:.2f}s)")
    
    last_m = print_mem_stage("6. After Waveform Read", last_m)
    
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    last_m = print_mem_stage("7. After gc.collect() & empty_cache()", last_m)
    
    print("\n" + "=" * 80)
    print(f"Total Process RSS at Finish: {last_m['rss_mb']:.2f} MB ({last_m['rss_mb']/1024:.2f} GB)")
    print(f"System Memory Used:          {last_m['sys_used_gb']:.2f} GB / 16.00 GB")
    print("=" * 80)

if __name__ == "__main__":
    main()
