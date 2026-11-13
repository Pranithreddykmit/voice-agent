"""
Test MMS-1B-ALL (Meta Massively Multilingual Speech) on Telugu benchmark.
Uses Wav2Vec2ForCTC with 'tel' language adapter.
"""

import sys
import gc
import json
import time
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForCTC, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ai-service"))

TESTSET_DIR = REPO_ROOT / "validation_artifacts" / "benchmark_test_set"
MANIFEST_PATH = TESTSET_DIR / "manifest.json"

from scripts.run_asr_model_benchmark import get_mem, compute_wer, compute_cer, categorize_errors

def run_mms_benchmark():
    print("\n" + "="*80)
    print("RUNNING BENCHMARK: 3. Facebook MMS-1B-ALL (Telugu Adapter)")
    print("="*80)
    
    mem_before = get_mem()
    print(f"Memory Before: RSS={mem_before['rss_mb']}MB | System Used={mem_before['sys_used_gb']}GB")
    
    model_id = "facebook/mms-1b-all"
    t0 = time.perf_counter()
    
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.set_target_lang("tel")
    
    model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang="tel", ignore_mismatched_sizes=True)
    model.eval()
    
    load_ms = (time.perf_counter() - t0) * 1000.0
    mem_loaded = get_mem()
    print(f"Model Loaded in {load_ms:.2f}ms | RSS={mem_loaded['rss_mb']}MB (Delta: {mem_loaded['rss_mb'] - mem_before['rss_mb']:.1f}MB)")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    # Test on the Telugu and code-mixed samples
    for sample in manifest:
        data, sr = sf.read(sample["wav_path"])
        if data.ndim > 1:
            data = data.mean(axis=1)
            
        t_inf_0 = time.perf_counter()
        inputs = processor(data, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            
        pred_ids = torch.argmax(logits, dim=-1)
        raw_transcript = processor.batch_decode(pred_ids)[0].strip()
        
        lat_ms = (time.perf_counter() - t_inf_0) * 1000.0
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
    results_mms = run_mms_benchmark()
    with open(REPO_ROOT / "validation_artifacts" / "results_mms_1b.json", "w", encoding="utf-8") as f:
        json.dump(results_mms, f, ensure_ascii=False, indent=2)
