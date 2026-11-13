#!/usr/bin/env python3
"""
Diagnostic script for single isolated TTS inference.
Measures memory before and after loading ONE model instance and synthesizing ONE short phrase.
"""

import os
import sys
import time
import psutil
import torch
import soundfile as sf
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-service")))

def get_mem():
    proc = psutil.Process()
    mem_info = proc.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    
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
        "mps_alloc_mb": mps_alloc_mb,
        "mps_driver_mb": mps_driver_mb,
        "sys_used_gb": vm.used / (1024**3),
        "sys_avail_gb": vm.available / (1024**3),
        "swap_used_gb": sm.used / (1024**3),
    }

def main():
    print("=" * 80, flush=True)
    print("  ISOLATED SINGLE TTS INFERENCE DIAGNOSTICS", flush=True)
    print("=" * 80, flush=True)
    
    m0 = get_mem()
    print(f"Initial Process RSS:  {m0['rss_mb']:.2f} MB | Sys Used: {m0['sys_used_gb']:.2f} GB | Swap: {m0['swap_used_gb']:.2f} GB", flush=True)
    
    # 1. Initialize ONE TTS Model Instance
    from app.speech.tts import IndicF5TTSProvider
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nInitializing 1 single IndicF5TTSProvider on device '{device}'...", flush=True)
    
    t0 = time.perf_counter()
    tts = IndicF5TTSProvider(adapter_path="models/indicf5_telugu_lora", device=device)
    init_ms = (time.perf_counter() - t0) * 1000.0
    
    m_loaded = get_mem()
    print(f"Model Loaded in {init_ms:.2f} ms | Adapter: {tts.adapter_name}", flush=True)
    print(f"Post-Load Process RSS: {m_loaded['rss_mb']:.2f} MB ({m_loaded['rss_mb']/1024:.2f} GB) | Delta: {m_loaded['rss_mb']-m0['rss_mb']:+.2f} MB", flush=True)
    print(f"MPS Allocated:         {m_loaded['mps_alloc_mb']:.2f} MB | Driver: {m_loaded['mps_driver_mb']:.2f} MB", flush=True)
    print(f"System Physical RAM:   {m_loaded['sys_used_gb']:.2f} GB / 16.00 GB", flush=True)
    
    # 2. Run ONE Short TTS Inference
    phrase = "నమస్కారం"
    print(f"\nSynthesizing ONE short phrase: \"{phrase}\"...", flush=True)
    
    t_synth = time.perf_counter()
    wav_bytes, lat_ms = tts.synthesize(phrase)
    synth_sec = time.perf_counter() - t_synth
    
    m_done = get_mem()
    data, sr = sf.read(io.BytesIO(wav_bytes))
    dur_sec = len(data) / float(sr)
    
    print(f"Synthesis Complete:    Latency={lat_ms:.2f} ms ({synth_sec:.2f} s) | Output Audio={dur_sec:.2f} s @ {sr} Hz", flush=True)
    print(f"Post-Synth RSS:        {m_done['rss_mb']:.2f} MB ({m_done['rss_mb']/1024:.2f} GB) | Delta: {m_done['rss_mb']-m_loaded['rss_mb']:+.2f} MB", flush=True)
    print(f"MPS Allocated / Driver:{m_done['mps_alloc_mb']:.2f} MB / {m_done['mps_driver_mb']:.2f} MB", flush=True)
    print(f"System Physical RAM:   {m_done['sys_used_gb']:.2f} GB / 16.00 GB", flush=True)
    print(f"System Swap Used:      {m_done['swap_used_gb']:.2f} GB", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
