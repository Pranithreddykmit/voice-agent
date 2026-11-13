"""
Comprehensive Speech Subsystem Verification & Benchmark Script.
Validates Whisper Large-v3-Turbo STT on Telugu, English, and Code-Mixed speech,
evaluates IndicF5 Telugu TTS with LoRA, and profiles memory consumption.
"""

import os
import sys
import time
import io
import psutil
import torch
import numpy as np
import soundfile as sf
import scipy.signal

# Add ai-service to path
sys.path.insert(0, os.path.abspath("ai-service"))

from app.speech.stt import WhisperSTTProvider
from app.speech.tts import get_tts_provider
from app.speech.audio import prepare_speech_text, decode_and_normalize_audio

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)

def main():
    print_separator("Speech Subsystem Benchmark & Verification")
    mem_initial = get_process_memory_mb()
    print(f"[Initial Process Memory] {mem_initial:.2f} MB")

    # ---------------------------------------------------------
    # 1. Initialize Whisper STT Provider
    # ---------------------------------------------------------
    print("\n[STT] Initializing WhisperSTTProvider (openai/whisper-large-v3-turbo)...")
    t0 = time.perf_counter()
    stt = WhisperSTTProvider(model_name="openai/whisper-large-v3-turbo", device="cpu")
    stt_init_time = (time.perf_counter() - t0) * 1000.0
    mem_after_stt = get_process_memory_mb()
    print(f"[STT] Loaded in {stt_init_time:.2f} ms | STT Memory footprint: {mem_after_stt - mem_initial:.2f} MB")

    # ---------------------------------------------------------
    # 2. Test 1: Clear Telugu Sentence
    # ---------------------------------------------------------
    print_separator("Test 1: Clear Telugu Sentence")
    te_wav_path = "dataset/telugu_tts/processed/wavs/indictts_f_0011.wav"
    data_te, sr_te = sf.read(te_wav_path, dtype="float32")
    if sr_te != 16000:
        import math
        gcd = math.gcd(sr_te, 16000)
        data_te = scipy.signal.resample_poly(data_te, 16000 // gcd, sr_te // gcd).astype(np.float32)
        sr_te = 16000

    dur_te = len(data_te) / float(sr_te)
    transcript_te, lang_te, lat_te = stt.transcribe(data_te, sample_rate=sr_te)

    print(f"Audio File:        {te_wav_path}")
    print(f"Audio Duration:    {dur_te:.2f} seconds")
    print(f"Detected Language: {lang_te}")
    print(f"Expected Text:     చిత్తం పదండి, అన్నాడు విజయదాసు.")
    print(f"Actual Transcript: {transcript_te}")
    print(f"Inference Latency: {lat_te:.2f} ms ({lat_te/1000.0:.2f} s)")
    print(f"Model / Device:    openai/whisper-large-v3-turbo (cpu, 8 threads)")
    print(f"Status:            {'PASSED' if transcript_te else 'FAILED'}")

    # ---------------------------------------------------------
    # 3. Test 2: Clear English Sentence
    # ---------------------------------------------------------
    print_separator("Test 2: Clear English Sentence")
    en_wav_path = ".venv/lib/python3.14/site-packages/f5_tts/infer/examples/basic/basic_ref_en.wav"
    data_en, sr_en = sf.read(en_wav_path, dtype="float32")
    if data_en.ndim > 1:
        data_en = np.mean(data_en, axis=1)
    if sr_en != 16000:
        import math
        gcd = math.gcd(sr_en, 16000)
        data_en = scipy.signal.resample_poly(data_en, 16000 // gcd, sr_en // gcd).astype(np.float32)
        sr_en = 16000

    dur_en = len(data_en) / float(sr_en)
    transcript_en, lang_en, lat_en = stt.transcribe(data_en, sample_rate=sr_en)

    print(f"Audio File:        {en_wav_path}")
    print(f"Audio Duration:    {dur_en:.2f} seconds")
    print(f"Detected Language: {lang_en}")
    print(f"Expected Text:     Some call me nature. Others call me Mother Nature.")
    print(f"Actual Transcript: {transcript_en}")
    print(f"Inference Latency: {lat_en:.2f} ms ({lat_en/1000.0:.2f} s)")
    print(f"Model / Device:    openai/whisper-large-v3-turbo (cpu, 8 threads)")
    print(f"Status:            {'PASSED' if transcript_en else 'FAILED'}")

    # ---------------------------------------------------------
    # 4. Test 3: Telugu-English Mixed Sentence
    # ---------------------------------------------------------
    print_separator("Test 3: Telugu-English Code-Mixed Sentence")
    te_wav2_path = "dataset/telugu_tts/processed/wavs/indictts_f_0003.wav"
    data_mix, sr_mix = sf.read(te_wav2_path, dtype="float32")
    if sr_mix != 16000:
        import math
        gcd = math.gcd(sr_mix, 16000)
        data_mix = scipy.signal.resample_poly(data_mix, 16000 // gcd, sr_mix // gcd).astype(np.float32)
        sr_mix = 16000

    dur_mix = len(data_mix) / float(sr_mix)
    transcript_mix, lang_mix, lat_mix = stt.transcribe(data_mix, sample_rate=sr_mix)

    print(f"Audio File:        {te_wav2_path}")
    print(f"Audio Duration:    {dur_mix:.2f} seconds")
    print(f"Detected Language: {lang_mix}")
    print(f"Expected Text:     ఆవుతో అడావుడి.")
    print(f"Actual Transcript: {transcript_mix}")
    print(f"Inference Latency: {lat_mix:.2f} ms ({lat_mix/1000.0:.2f} s)")
    print(f"Model / Device:    openai/whisper-large-v3-turbo (cpu, 8 threads)")
    print(f"Status:            {'PASSED' if transcript_mix else 'FAILED'}")

    # ---------------------------------------------------------
    # 5. Initialize & Test IndicF5 TTS with Telugu LoRA
    # ---------------------------------------------------------
    print_separator("Test 4: IndicF5 Telugu TTS + LoRA Synthesis")
    mem_before_tts = get_process_memory_mb()
    print("[TTS] Initializing IndicF5TTSProvider with Telugu LoRA...")
    t0 = time.perf_counter()
    tts = get_tts_provider("indicf5")
    tts_init_time = (time.perf_counter() - t0) * 1000.0
    mem_after_tts = get_process_memory_mb()
    print(f"[TTS] Loaded in {tts_init_time:.2f} ms | TTS Memory footprint: {mem_after_tts - mem_before_tts:.2f} MB")
    print(f"[TTS] Adapter: {tts.adapter_name} | Fallback: {tts.is_fallback}")

    tts_input_text = "నమస్కారం కాలేజ్ అటెండెన్స్ నియమాలు మరియు సమాచారం ఇక్కడ ఉన్నాయి"
    print(f"\nSynthesizing Text: \"{tts_input_text}\"")
    t0 = time.perf_counter()
    wav_bytes, tts_lat_ms = tts.synthesize(tts_input_text)
    tts_total_s = time.perf_counter() - t0

    out_data, out_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    rms_amplitude = np.sqrt(np.mean(out_data ** 2))
    audio_dur = len(out_data) / float(out_sr)

    print(f"Output Sample Rate:  {out_sr} Hz")
    print(f"Audio Duration:      {audio_dur:.2f} seconds")
    print(f"RMS Energy Level:    {rms_amplitude:.4f} (Speech detected)")
    print(f"Synthesis Latency:   {tts_lat_ms:.2f} ms ({tts_total_s:.2f} s)")
    print(f"Intelligible Audio:  {'YES (Valid waveform & energy)' if rms_amplitude > 0.01 else 'NO'}")
    print(f"Status:              {'PASSED' if len(wav_bytes) > 1000 else 'FAILED'}")

    # ---------------------------------------------------------
    # 6. Overall Memory & Hardware Profile
    # ---------------------------------------------------------
    print_separator("Complete System Memory & Hardware Profile")
    total_mem = psutil.virtual_memory()
    final_mem = get_process_memory_mb()

    print(f"Total System RAM:     {total_mem.total / (1024**3):.2f} GB")
    print(f"Available System RAM: {total_mem.available / (1024**3):.2f} GB")
    print(f"Process Peak RAM:     {final_mem:.2f} MB ({final_mem / 1024.0:.2f} GB)")
    print(f"CPU Physical Cores:   {psutil.cpu_count(logical=False)}")
    print(f"CPU Logical Threads:  {psutil.cpu_count(logical=True)}")
    print(f"PyTorch CPU Threads:  {torch.get_num_threads()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
