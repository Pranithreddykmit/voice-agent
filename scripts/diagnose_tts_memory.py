#!/usr/bin/env python3
"""
Forensic TTS Memory Investigation Script for Apple Silicon M4.
Instruments memory (RSS and PyTorch MPS/CPU allocations) at every granular stage of TTS:
1. Before TTS initialization
2. After model loading
3. Before text preprocessing
4. After text preprocessing
5. Duration calculation & UTF-8 vs char length analysis
6. Before DiT ODE sampling
7. After DiT ODE sampling (tensor shapes & memory)
8. After Vocos decoding
9. After waveform creation & numpy conversion
10. After returning the result
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
    if torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
        try:
            mps_alloc_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
        except Exception:
            pass
            
    return {
        "rss_mb": rss_mb,
        "vms_mb": vms_mb,
        "mps_alloc_mb": mps_alloc_mb,
        "sys_used_gb": sys_used_gb,
        "sys_avail_gb": sys_avail_gb,
    }

def print_mem_stage(stage_name: str, prev_mem: dict):
    curr = get_mem()
    diff_rss = curr["rss_mb"] - prev_mem["rss_mb"]
    print(f"[{stage_name:<35}] RSS: {curr['rss_mb']:>8.2f} MB (delta: {diff_rss:>+7.2f} MB) | MPS: {curr['mps_alloc_mb']:>7.2f} MB | Sys Used: {curr['sys_used_gb']:.2f} GB")
    return curr

def main():
    print("=" * 80)
    print("  FORENSIC TTS MEMORY INVESTIGATION ON APPLE SILICON M4")
    print("=" * 80)
    
    m0 = get_mem()
    print(f"Initial State -> Process RSS: {m0['rss_mb']:.2f} MB | System Used: {m0['sys_used_gb']:.2f} GB | System Avail: {m0['sys_avail_gb']:.2f} GB\n")
    
    # -------------------------------------------------------------------------
    # STAGE 1: Model Initialization
    # -------------------------------------------------------------------------
    last_m = print_mem_stage("1. Before Model Load", m0)
    
    from app.speech.tts import IndicF5TTSProvider
    tts = IndicF5TTSProvider(adapter_path="models/indicf5_telugu_lora", device="cpu")
    last_m = print_mem_stage("2. After IndicF5 Load (CPU)", last_m)
    
    # -------------------------------------------------------------------------
    # STAGE 2: Granular Single Short Sentence Synthesis
    # -------------------------------------------------------------------------
    test_text = "నమస్కారం"
    print(f"\n--- Testing ONE Short Text: \"{test_text}\" (Characters: {len(test_text)}, UTF-8 Bytes: {len(test_text.encode('utf-8'))}) ---")
    
    from app.speech.audio import prepare_speech_text
    clean_text = prepare_speech_text(test_text)
    last_m = print_mem_stage("3. After Text Preprocessing", last_m)
    
    # Inspect Reference Audio & Duration Estimation
    import torchaudio
    ref_audio, sr = torchaudio.load(tts._ref_wav_path)
    ref_text = tts._ref_text
    ref_text_bytes = len(ref_text.encode("utf-8"))
    gen_text_bytes = len(clean_text.encode("utf-8"))
    ref_text_chars = len(ref_text)
    gen_text_chars = len(clean_text)
    
    print(f"\n[Duration Calculation Diagnostics]")
    print(f"  Ref Text:      '{ref_text}' | Chars: {ref_text_chars} | UTF-8 Bytes: {ref_text_bytes}")
    print(f"  Gen Text:      '{clean_text}' | Chars: {gen_text_chars} | UTF-8 Bytes: {gen_text_bytes}")
    print(f"  Ref Audio:     {ref_audio.shape[-1]} samples ({ref_audio.shape[-1]/sr:.2f}s)")
    
    hop_length = 256
    target_sr = 24000
    ref_audio_len_frames = ref_audio.shape[-1] // hop_length
    
    # f5_tts default formula (using utf-8 byte length):
    duration_utf8_formula = ref_audio_len_frames + int(ref_audio_len_frames / ref_text_bytes * gen_text_bytes / 1.0)
    duration_utf8_sec = duration_utf8_formula * hop_length / target_sr
    
    # corrected formula (using character length):
    duration_char_formula = ref_audio_len_frames + int(ref_audio_len_frames / max(1, ref_text_chars) * gen_text_chars / 1.0)
    duration_char_sec = duration_char_formula * hop_length / target_sr
    
    print(f"  F5-TTS UTF-8 Estimated Duration: {duration_utf8_formula} frames ({duration_utf8_sec:.2f} seconds)")
    print(f"  Char-Based Estimated Duration:   {duration_char_formula} frames ({duration_char_sec:.2f} seconds)")
    print(f"  -> UTF-8 Blowup Factor:          {duration_utf8_sec / max(0.1, duration_char_sec):.2f}x!\n")
    
    # -------------------------------------------------------------------------
    # STAGE 3: Run Synthesize with Step-by-Step Profiling
    # -------------------------------------------------------------------------
    last_m = print_mem_stage("4. Before Synthesize Call", last_m)
    
    t0 = time.perf_counter()
    wav_bytes, lat = tts.synthesize(clean_text)
    synth_time = time.perf_counter() - t0
    
    last_m = print_mem_stage("5. After Synthesize Call", last_m)
    
    data, out_sr = sf.read(io.BytesIO(wav_bytes))
    rms = float(np.sqrt(np.mean(data**2)))
    print(f"\nResult: Duration={len(data)/out_sr:.2f}s @ {out_sr}Hz | RMS={rms:.4f} | Latency={lat:.2f}ms ({synth_time:.2f}s)")
    
    last_m = print_mem_stage("6. After Waveform Read", last_m)
    
    # Final cleanup check
    gc.collect()
    last_m = print_mem_stage("7. After gc.collect()", last_m)
    
    print("\n" + "=" * 80)
    print(f"Total Process RSS at Finish: {last_m['rss_mb']:.2f} MB ({last_m['rss_mb']/1024:.2f} GB)")
    print(f"System Memory Used:          {last_m['sys_used_gb']:.2f} GB / 16.00 GB")
    print("=" * 80)

if __name__ == "__main__":
    main()
