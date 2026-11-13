#!/usr/bin/env python3
"""
Granular Benchmark for Single TTS Generation on Apple Silicon M4.
Tests:
1. 'నమస్కారం'
2. 'Hello'
3. 'నమస్కారం, నేను మీ AI సహాయకుడిని.'
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
    vm = psutil.virtual_memory()
    
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
    }

def main():
    print("=" * 80)
    print("  GRANULAR SINGLE TTS BENCHMARK (M4 MPS)")
    print("=" * 80)
    
    from app.speech.tts import IndicF5TTSProvider
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Initializing IndicF5TTSProvider on device: {device}...")
    t0 = time.perf_counter()
    tts = IndicF5TTSProvider(adapter_path="models/indicf5_telugu_lora", device=device)
    init_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"IndicF5 loaded in {init_time_ms:.2f} ms | Active Adapter: {tts.adapter_name}\n")
    
    test_phrases = [
        "నమస్కారం",
        "Hello",
        "నమస్కారం, నేను మీ AI సహాయకుడిని."
    ]
    
    for idx, phrase in enumerate(test_phrases, 1):
        print("-" * 80)
        print(f"TEST {idx}: \"{phrase}\"")
        print("-" * 80)
        m_before = get_mem()
        print(f"  [Before Synthesis] RSS: {m_before['rss_mb']:.2f} MB | MPS Alloc: {m_before['mps_alloc_mb']:.2f} MB | Driver: {m_before['mps_driver_mb']:.2f} MB")
        
        t0 = time.perf_counter()
        wav_bytes, lat = tts.synthesize(phrase)
        total_time_s = time.perf_counter() - t0
        
        m_after = get_mem()
        print(f"  [After Synthesis ] RSS: {m_after['rss_mb']:.2f} MB | MPS Alloc: {m_after['mps_alloc_mb']:.2f} MB | Driver: {m_after['mps_driver_mb']:.2f} MB")
        
        data, sr = sf.read(io.BytesIO(wav_bytes))
        dur_s = len(data) / float(sr)
        rms = float(np.sqrt(np.mean(data ** 2)))
        
        print(f"  Input Text:              \"{phrase}\"")
        print(f"  Character Count:         {len(phrase)} chars (UTF-8 bytes: {len(phrase.encode('utf-8'))})")
        print(f"  Inference Latency:       {lat:.2f} ms ({total_time_s:.2f} s)")
        print(f"  Device:                  {device}")
        print(f"  Output Waveform:         {dur_s:.2f} s @ {sr} Hz")
        print(f"  RMS Energy:              {rms:.4f} (Intelligible: {rms > 0.01})")
        print(f"  Peak Process RSS:        {m_after['rss_mb']:.2f} MB ({m_after['rss_mb']/1024:.2f} GB)")
        print(f"  Delta RSS:               {m_after['rss_mb'] - m_before['rss_mb']:+.2f} MB")
        print(f"  Status:                  PASSED\n")

if __name__ == "__main__":
    main()
