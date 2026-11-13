"""
Benchmark Candidate 2: Optimized Whisper Large-v3-Turbo
Tests Whisper Large-v3-Turbo with:
- Removed repetition penalty (restoring Telugu subwords)
- Prompt conditioning (initial_prompt for Telugu + English code-mixed academic vocabulary)
- Proper temperature fallback & native auto language detection
- Device benchmark (CPU vs MPS)
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
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ai-service"))

TESTSET_DIR = REPO_ROOT / "validation_artifacts" / "benchmark_test_set"
MANIFEST_PATH = TESTSET_DIR / "manifest.json"

from scripts.run_asr_model_benchmark import get_mem, compute_wer, compute_cer, categorize_errors

PROMPT_TEXT = "Semester exams, hostel fee, library card, attendance, medical leave, procedure, documents, నిబంధనలు, హాజరు, పరీక్షలు, కాలేజీ, లైబ్రరీ, ఫీజు, సెలవు, దరఖాస్తు, నమస్కారం."

def run_optimized_whisper_turbo(device_name: str = "cpu"):
    print("\n" + "="*80)
    print(f"RUNNING BENCHMARK: 2. Optimized Whisper Large-v3-Turbo (Device={device_name.upper()})")
    print("="*80)
    
    mem_before = get_mem()
    print(f"Memory Before: RSS={mem_before['rss_mb']}MB | System Used={mem_before['sys_used_gb']}GB")
    
    t_load_0 = time.perf_counter()
    torch_device = device_name
    dtype = torch.float32 if device_name in ["cpu", "mps"] else torch.float16
    
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    model.to(torch_device)
    model.eval()
    
    load_ms = (time.perf_counter() - t_load_0) * 1000.0
    mem_loaded = get_mem()
    print(f"Model Loaded on {torch_device} in {load_ms:.2f}ms | RSS={mem_loaded['rss_mb']}MB (Delta: {mem_loaded['rss_mb'] - mem_before['rss_mb']:.1f}MB)")

    # Prepare prompt token ids
    prompt_ids_raw = processor.get_prompt_ids(PROMPT_TEXT)
    prompt_ids = torch.from_numpy(prompt_ids_raw).to(torch_device) if isinstance(prompt_ids_raw, np.ndarray) else torch.tensor(prompt_ids_raw, device=torch_device)
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    for sample in manifest:
        data, sr = sf.read(sample["wav_path"])
        if data.ndim > 1:
            data = data.mean(axis=1)
            
        t0 = time.perf_counter()
        inputs = processor(data, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(torch_device, dtype=dtype)
        
        with torch.no_grad():
            out = model.generate(
                input_features,
                prompt_ids=prompt_ids,
                task="transcribe"
            )
            
        raw_transcript = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        lat_sec = lat_ms / 1000.0
        dur_sec = sample["duration_sec"]
        rtf = round(lat_sec / dur_sec, 3) if dur_sec > 0 else 0.0

        wer = compute_wer(sample["ref_text"], raw_transcript)
        cer = compute_cer(sample["ref_text"], raw_transcript)
        err_tags = categorize_errors(sample["ref_text"], raw_transcript)

        res_entry = {
            "id": sample["id"],
            "category": sample["category"],
            "duration_sec": dur_sec,
            "latency_ms": round(lat_ms, 2),
            "rtf": rtf,
            "ref_text": sample["ref_text"],
            "transcript": raw_transcript,
            "wer": wer,
            "cer": cer,
            "errors": err_tags
        }
        results.append(res_entry)
        print(f"[{sample['id']}] ({dur_sec}s -> {lat_ms:.1f}ms, RTF={rtf:.2f}x)")
        print(f"   Ref:  {sample['ref_text']}")
        print(f"   Hyp:  {raw_transcript}")
        print(f"   WER:  {wer*100:.1f}% | CER: {cer*100:.1f}% | Tags: {', '.join(err_tags)}\n")

    # Clean unload
    del model
    del processor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    mem_after = get_mem()
    print(f"Memory After Unload: RSS={mem_after['rss_mb']}MB | System Used={mem_after['sys_used_gb']}GB")
    return results

if __name__ == "__main__":
    results_opt = run_optimized_whisper_turbo(device_name="cpu")
    with open(REPO_ROOT / "validation_artifacts" / "results_optimized_whisper_turbo.json", "w", encoding="utf-8") as f:
        json.dump(results_opt, f, ensure_ascii=False, indent=2)
