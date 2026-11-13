"""
Benchmark Candidate: OpenAI Whisper-Small (241M parameters)
Tests lightweight Whisper architecture on Telugu / English / Code-mixed test set.
"""

import sys
import gc
import json
import time
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

def run_whisper_small():
    print("\n" + "="*80)
    print("RUNNING BENCHMARK: OpenAI Whisper-Small (241M params)")
    print("="*80)
    
    mem_before = get_mem()
    print(f"Memory Before: RSS={mem_before['rss_mb']}MB | System Used={mem_before['sys_used_gb']}GB")
    
    model_id = "openai/whisper-small"
    t_load_0 = time.perf_counter()
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch.float32)
    model.to("cpu")
    model.eval()
    
    load_ms = (time.perf_counter() - t_load_0) * 1000.0
    mem_loaded = get_mem()
    print(f"Model Loaded in {load_ms:.2f}ms | RSS={mem_loaded['rss_mb']}MB (Delta: {mem_loaded['rss_mb'] - mem_before['rss_mb']:.1f}MB)")

    prompt_ids_raw = processor.get_prompt_ids(PROMPT_TEXT)
    prompt_ids = torch.from_numpy(prompt_ids_raw) if isinstance(prompt_ids_raw, np.ndarray) else torch.tensor(prompt_ids_raw)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    for sample in manifest:
        data, sr = sf.read(sample["wav_path"])
        if data.ndim > 1:
            data = data.mean(axis=1)
            
        t0 = time.perf_counter()
        inputs = processor(data, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to("cpu")
        
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
    mem_after = get_mem()
    print(f"Memory After Unload: RSS={mem_after['rss_mb']}MB | System Used={mem_after['sys_used_gb']}GB")
    return results

if __name__ == "__main__":
    results = run_whisper_small()
    with open(REPO_ROOT / "validation_artifacts" / "results_whisper_small.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
