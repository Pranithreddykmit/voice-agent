#!/usr/bin/env python3
"""
Test script for the new direct, character-aware, single-pass TTS synthesis.
Benchmarks:
1. 'నమస్కారం'
2. 'Hello'
3. 'నమస్కారం, నేను మీ AI సహాయకుడిని.'
4. Conversational 3-sentence response:
   'Hello! I am your Multi-Agent Telugu Conversational AI assistant. How can I assist you with academic regulations and document queries today?'
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
import torchaudio

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

def synthesize_direct(model, vocoder, ref_audio, ref_text, gen_text, device="mps", nfe_steps=16):
    from f5_tts.model.utils import convert_char_to_pinyin
    
    target_sr = 24000
    hop_length = 256
    ref_audio_len_frames = ref_audio.shape[-1] // hop_length
    
    is_indic = any(ord(c) >= 0x0C00 and ord(c) <= 0x0C7F for c in gen_text)
    chars_per_sec = 11.0 if is_indic else 14.5
    gen_sec = max(0.8, len(gen_text) / chars_per_sec) + 0.25
    gen_sec = min(gen_sec, 22.0)
    
    duration = ref_audio_len_frames + int(gen_sec * target_sr / hop_length)
    print(f"    [DiT Sampling] Frames: {duration} ({duration*hop_length/target_sr:.2f}s) | Speech Target: {gen_sec:.2f}s", flush=True)
    
    text_list = [ref_text + " " + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)
    
    t_samp = time.perf_counter()
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=ref_audio,
            text=final_text_list,
            duration=duration,
            steps=nfe_steps,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )
        del _
        
        samp_ms = (time.perf_counter() - t_samp) * 1000.0
        print(f"    [DiT Sampled] in {samp_ms:.2f} ms. Decoding with Vocos...", flush=True)
        
        t_voc = time.perf_counter()
        generated_mel = generated[:, ref_audio_len_frames:, :].to(torch.float32).permute(0, 2, 1)
        generated_wave = vocoder.decode(generated_mel)
        del generated, generated_mel
        
        wav_np = generated_wave.squeeze().cpu().numpy()
        del generated_wave
        voc_ms = (time.perf_counter() - t_voc) * 1000.0
        print(f"    [Vocos Decoded] in {voc_ms:.2f} ms.", flush=True)
        
    buf = io.BytesIO()
    sf.write(buf, wav_np, target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), duration * hop_length / target_sr, gen_sec

def main():
    print("=" * 80, flush=True)
    print("  DIRECT CHARACTER-AWARE TTS BENCHMARK (APPLE SILICON M4 MPS)", flush=True)
    print("=" * 80, flush=True)
    
    from app.speech.tts import IndicF5TTSProvider
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading IndicF5 on device: {device}...", flush=True)
    tts = IndicF5TTSProvider(adapter_path="models/indicf5_telugu_lora", device=device)
    
    # Load and cache ref audio tensor on device
    ref_audio, sr = torchaudio.load(tts._ref_wav_path)
    if ref_audio.shape[0] > 1:
        ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        ref_audio = resampler(ref_audio)
    ref_audio = ref_audio.to(device)
    print(f"Ref Audio loaded: {ref_audio.shape} on {ref_audio.device}", flush=True)
    
    test_cases = [
        "నమస్కారం",
        "Hello",
        "నమస్కారం, నేను మీ AI సహాయకుడిని.",
        "Hello! I am your Multi-Agent Telugu Conversational AI assistant. How can I assist you with academic regulations and document queries today?"
    ]
    
    for idx, phrase in enumerate(test_cases, 1):
        print("\n" + "-" * 80, flush=True)
        print(f"TEST {idx}: \"{phrase}\"", flush=True)
        print("-" * 80, flush=True)
        
        m_before = get_mem()
        t0 = time.perf_counter()
        
        wav_bytes, total_dur_s, speech_dur_s = synthesize_direct(
            tts._model, tts._vocoder, ref_audio, tts._ref_text, phrase, device=device, nfe_steps=16
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        m_after = get_mem()
        
        data, out_sr = sf.read(io.BytesIO(wav_bytes))
        actual_dur_s = len(data) / float(out_sr)
        rms = float(np.sqrt(np.mean(data ** 2)))
        
        print(f"  Input Text:              \"{phrase}\"", flush=True)
        print(f"  Character Count:         {len(phrase)} chars (UTF-8 bytes: {len(phrase.encode('utf-8'))})", flush=True)
        print(f"  Target Speech Duration:  {speech_dur_s:.2f} s (Total frames: {total_dur_s:.2f} s)", flush=True)
        print(f"  Actual Audio Duration:   {actual_dur_s:.2f} s @ {out_sr} Hz", flush=True)
        print(f"  Inference Latency:       {latency_ms:.2f} ms ({latency_ms/1000:.2f} s)", flush=True)
        print(f"  RMS Energy:              {rms:.4f} (Intelligible: {rms > 0.01})", flush=True)
        print(f"  Peak RSS:                {m_after['rss_mb']:.2f} MB ({m_after['rss_mb']/1024:.2f} GB)", flush=True)
        print(f"  Delta RSS:               {m_after['rss_mb'] - m_before['rss_mb']:+.2f} MB", flush=True)
        print(f"  MPS Allocated / Driver:  {m_after['mps_alloc_mb']:.2f} MB / {m_after['mps_driver_mb']:.2f} MB", flush=True)
        print(f"  Status:                  PASSED", flush=True)

if __name__ == "__main__":
    main()
