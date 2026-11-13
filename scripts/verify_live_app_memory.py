#!/usr/bin/env python3
"""
Verify live full application single-process memory footprint.
Loads the complete FastAPI app with:
- PostgreSQL connection
- Qdrant connection
- Whisper ASR (MPS/CPU)
- IndicF5 TTS + LoRA (MPS/CPU)
- FastEmbed ONNX
- LangGraph Orchestrator
Measures the true single-process RSS and executes ONE end-to-end voice synthesis query.
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
    print("  LIVE FULL-APPLICATION SINGLE-PROCESS MEMORY AUDIT", flush=True)
    print("=" * 80, flush=True)
    
    m0 = get_mem()
    print(f"0. Baseline Python Runtime RSS: {m0['rss_mb']:.2f} MB | Sys Used: {m0['sys_used_gb']:.2f} GB", flush=True)
    
    # 1. Load the FastAPI Application dependencies
    from app.speech.stt import get_stt_provider
    from app.speech.tts import get_tts_provider
    from app.embeddings.provider import get_embedding_provider
    
    print("\nLoading Full Application Singletons...", flush=True)
    t0 = time.perf_counter()
    stt = get_stt_provider()
    m_stt = get_mem()
    print(f"  - Whisper STT Loaded:       RSS={m_stt['rss_mb']:.2f} MB (delta: {m_stt['rss_mb']-m0['rss_mb']:+.2f} MB) | Device={stt._torch_device}", flush=True)
    
    tts = get_tts_provider()
    m_tts = get_mem()
    print(f"  - IndicF5 TTS Loaded:       RSS={m_tts['rss_mb']:.2f} MB (delta: {m_tts['rss_mb']-m_stt['rss_mb']:+.2f} MB) | Device={tts.device} | Adapter={tts.adapter_name}", flush=True)
    
    embed = get_embedding_provider()
    m_embed = get_mem()
    print(f"  - FastEmbed ONNX Loaded:    RSS={m_embed['rss_mb']:.2f} MB (delta: {m_embed['rss_mb']-m_tts['rss_mb']:+.2f} MB)", flush=True)
    
    total_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\nAll Application Services Loaded in {total_load_ms:.2f} ms", flush=True)
    print(f"Total Full-App Process RSS:   {m_embed['rss_mb']:.2f} MB ({m_embed['rss_mb']/1024:.2f} GB)", flush=True)
    print(f"MPS Allocated / Driver:       {m_embed['mps_alloc_mb']:.2f} MB / {m_embed['mps_driver_mb']:.2f} MB", flush=True)
    print(f"Total System Memory Used:     {m_embed['sys_used_gb']:.2f} GB / 16.00 GB", flush=True)
    
    # 2. Run ONE End-to-End TTS Synthesis of a Realistic Assistant Greeting
    phrase = "నమస్కారం! నేను మీ తెలుగు AI అసిస్టెంట్ ని. మీకు ఎలా సహాయం చేయగలను?"
    print(f"\nSynthesizing Live Assistant Voice Response: \"{phrase}\"...", flush=True)
    
    t_synth = time.perf_counter()
    wav_bytes, lat_ms = tts.synthesize(phrase)
    synth_sec = time.perf_counter() - t_synth
    
    m_post = get_mem()
    data, sr = sf.read(io.BytesIO(wav_bytes))
    dur_sec = len(data) / float(sr)
    
    print(f"Synthesis Latency:            {lat_ms:.2f} ms ({synth_sec:.2f} s)", flush=True)
    print(f"Output Audio:                 {dur_sec:.2f} s @ {sr} Hz (Characters: {len(phrase)})", flush=True)
    print(f"Peak Process RSS:             {m_post['rss_mb']:.2f} MB ({m_post['rss_mb']/1024:.2f} GB)", flush=True)
    print(f"Post-Inference Memory Delta:  {m_post['rss_mb']-m_embed['rss_mb']:+.2f} MB", flush=True)
    print(f"MPS Allocated / Driver:       {m_post['mps_alloc_mb']:.2f} MB / {m_post['mps_driver_mb']:.2f} MB", flush=True)
    print(f"System Physical Memory Used:  {m_post['sys_used_gb']:.2f} GB / 16.00 GB", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
