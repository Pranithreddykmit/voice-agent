"""
ASR Model Investigation & Benchmark Runner.
Executes isolated benchmarks ONE MODEL AT A TIME with strict memory tracking.
Computes:
- Audio duration
- Inference latency (ms) & Real-Time Factor (RTF)
- Word Error Rate (WER) & Character Error Rate (CER)
- Qualitative error categorization (Devanagari bleed, Romanized Telugu, hallucination, etc.)
- Peak RSS RAM & System Memory before/after
"""

import os
import sys
import gc
import json
import time
import psutil
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ai-service"))

TESTSET_DIR = REPO_ROOT / "validation_artifacts" / "benchmark_test_set"
MANIFEST_PATH = TESTSET_DIR / "manifest.json"

def get_mem():
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    vm = psutil.virtual_memory()
    return {
        "rss_mb": round(rss_mb, 2),
        "sys_used_gb": round(vm.used / (1024**3), 2),
        "sys_avail_gb": round(vm.available / (1024**3), 2)
    }

def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate using Levenshtein distance."""
    ref = ref.strip().replace(" ", "")
    hyp = hyp.strip().replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    r_len, h_len = len(ref), len(hyp)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return round(dp[r_len][h_len] / float(r_len), 4)

def compute_wer(ref: str, hyp: str) -> float:
    """Word Error Rate."""
    import re
    r_words = re.findall(r"\w+", ref.lower())
    h_words = re.findall(r"\w+", hyp.lower())
    if not r_words:
        return 0.0 if not h_words else 1.0
    r_len, h_len = len(r_words), len(h_words)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if r_words[i - 1] == h_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return round(dp[r_len][h_len] / float(r_len), 4)

def categorize_errors(ref: str, hyp: str) -> list:
    """Classifies linguistic error patterns."""
    errors = []
    # 1. Devanagari script bleed (Unicode \u0900-\u097F)
    if any('\u0900' <= c <= '\u097F' for c in hyp):
        errors.append("Devanagari script bleed")
    
    # 2. Tamil script bleed (\u0B80-\u0BFF)
    if any('\u0B80' <= c <= '\u0BFF' for c in hyp):
        errors.append("Tamil script bleed")
        
    # 3. Kannada script bleed (\u0C80-\u0CFF)
    if any('\u0C80' <= c <= '\u0CFF' for c in hyp):
        errors.append("Kannada script bleed")

    # 4. Check if Telugu reference was Romanized
    has_telugu_ref = any('\u0C00' <= c <= '\u0C7F' for c in ref)
    has_telugu_hyp = any('\u0C00' <= c <= '\u0C7F' for c in hyp)
    if has_telugu_ref and not has_telugu_hyp and any(c.isalpha() for c in hyp):
        errors.append("Romanized Telugu")

    # 5. Severe truncation / omitted words
    if len(hyp.strip()) < len(ref.strip()) * 0.3 and len(ref.strip()) > 10:
        errors.append("Omitted words / Truncation")

    # 6. Hallucination / repetition
    if len(hyp.strip()) > len(ref.strip()) * 2.5:
        errors.append("Hallucination / Token explosion")

    if not errors and compute_wer(ref, hyp) > 0.4:
        errors.append("Phonetic/Subword mismatch")

    if not errors:
        errors.append("Clean / Faithful")

    return errors


# =========================================================================
# MODEL 1: Current Whisper Turbo Baseline (As configured in app/speech/stt.py)
# =========================================================================
def run_current_whisper_turbo_baseline():
    print("\n" + "="*80)
    print("RUNNING BENCHMARK: 1. Current Whisper Large-v3-Turbo Baseline")
    print("="*80)
    
    mem_before = get_mem()
    print(f"Memory Before: RSS={mem_before['rss_mb']}MB | System Used={mem_before['sys_used_gb']}GB")
    
    from app.speech.stt import WhisperSTTProvider
    stt = WhisperSTTProvider(model_name="openai/whisper-large-v3-turbo")
    
    mem_loaded = get_mem()
    print(f"Memory Loaded: RSS={mem_loaded['rss_mb']}MB (Model size delta: {mem_loaded['rss_mb'] - mem_before['rss_mb']:.1f}MB)")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    for sample in manifest:
        data, sr = sf.read(sample["wav_path"])
        if data.ndim > 1:
            data = data.mean(axis=1)
            
        t0 = time.perf_counter()
        transcript, lang, lat_ms = stt.transcribe(data.astype(np.float32), sample_rate=sr)
        lat_sec = lat_ms / 1000.0
        dur_sec = sample["duration_sec"]
        rtf = round(lat_sec / dur_sec, 3) if dur_sec > 0 else 0.0

        wer = compute_wer(sample["ref_text"], transcript)
        cer = compute_cer(sample["ref_text"], transcript)
        err_tags = categorize_errors(sample["ref_text"], transcript)

        res_entry = {
            "id": sample["id"],
            "category": sample["category"],
            "duration_sec": dur_sec,
            "latency_ms": lat_ms,
            "rtf": rtf,
            "detected_lang": lang,
            "ref_text": sample["ref_text"],
            "transcript": transcript,
            "wer": wer,
            "cer": cer,
            "errors": err_tags
        }
        results.append(res_entry)
        print(f"[{sample['id']}] ({dur_sec}s -> {lat_ms:.1f}ms, RTF={rtf:.2f}x, Lang={lang})")
        print(f"   Ref:  {sample['ref_text']}")
        print(f"   Hyp:  {transcript}")
        print(f"   WER:  {wer*100:.1f}% | CER: {cer*100:.1f}% | Tags: {', '.join(err_tags)}\n")

    # Clean unload
    del stt
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    mem_after = get_mem()
    print(f"Memory After Unload: RSS={mem_after['rss_mb']}MB | System Used={mem_after['sys_used_gb']}GB")
    return results

if __name__ == "__main__":
    results_baseline = run_current_whisper_turbo_baseline()
    with open(REPO_ROOT / "validation_artifacts" / "results_baseline_whisper_turbo.json", "w", encoding="utf-8") as f:
        json.dump(results_baseline, f, ensure_ascii=False, indent=2)
