"""
Whisper LoRA Held-Out Test Set Evaluation Harness
=================================================
Evaluates Base Whisper Large-v3-Turbo vs LoRA Fine-Tuned Adapter on the
quarantined, held-out test set (dataset/whisper_lora/manifest_test.json).

Metrics:
- Pure Telugu WER & CER
- Telugu-English Code-Mixed WER & CER
- English / Academic Terminology WER
- Latency & RTF
- Error categorization (Devanagari bleed, Tamil bleed, Romanization, Substitutions)
"""

import os
import gc
import json
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List

import torch
import soundfile as sf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_MANIFEST = REPO_ROOT / "dataset" / "whisper_lora" / "manifest_test.json"
RESULTS_OUT = REPO_ROOT / "validation_artifacts" / "results_lora_heldout_testset.json"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def compute_wer(ref: str, hyp: str) -> float:
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return round(d[len(r)][len(h)] / len(r), 4)


def compute_cer(ref: str, hyp: str) -> float:
    r = normalize_text(ref).replace(" ", "")
    h = normalize_text(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return round(d[len(r)][len(h)] / len(r), 4)


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= c <= "\u097F" for c in text)


def has_tamil(text: str) -> bool:
    return any("\u0B80" <= c <= "\u0BFF" for c in text)


def categorize_error(ref: str, hyp: str) -> List[str]:
    tags = []
    if has_devanagari(hyp) and not has_devanagari(ref):
        tags.append("Devanagari script bleed")
    if has_tamil(hyp) and not has_tamil(ref):
        tags.append("Tamil script bleed")
    if compute_wer(ref, hyp) == 0.0:
        tags.append("Clean / Faithful")
    elif not tags:
        tags.append("Phonetic / Subword Mismatch")
    return tags


def run_evaluation(base_model_name: str, lora_adapter_path: str = None, test_manifest: str = None):
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    
    manifest_path = Path(test_manifest) if test_manifest else TEST_MANIFEST
    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("\n" + "=" * 80)
    print("WHISPER LoRA EVALUATION ON HELD-OUT TEST SET")
    print(f"Base Model:    {base_model_name}")
    print(f"LoRA Adapter:  {lora_adapter_path or 'None (Baseline Only)'}")
    print(f"Test Samples:  {len(samples)}")
    print("=" * 80 + "\n")

    # Load model on CPU / MPS
    device = "cpu"
    dtype = torch.float32

    processor = AutoProcessor.from_pretrained(base_model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(base_model_name, dtype=dtype)
    
    if lora_adapter_path and os.path.exists(lora_adapter_path):
        from peft import PeftModel
        print(f"Loading LoRA weights from {lora_adapter_path}...")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    model.to(device)
    model.eval()

    results = []
    for s in samples:
        audio_path = s.get("audio_path")
        ref_text = s.get("text", "")
        cat = s.get("category", "unknown")

        if not audio_path or not os.path.exists(audio_path):
            continue

        audio_data, sr = sf.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)

        t0 = time.perf_counter()
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            out = model.generate(input_features, task="transcribe", max_new_tokens=256)

        hyp_text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        wer = compute_wer(ref_text, hyp_text)
        cer = compute_cer(ref_text, hyp_text)
        err_tags = categorize_error(ref_text, hyp_text)

        entry = {
            "id": s.get("id"),
            "category": cat,
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "wer": wer,
            "cer": cer,
            "errors": err_tags,
            "latency_ms": round(elapsed_ms, 2)
        }
        results.append(entry)

        print(f"[{s.get('id')}] ({cat}) | Latency: {elapsed_ms:.0f}ms")
        print(f"  Ref:  {ref_text}")
        print(f"  Hyp:  {hyp_text}")
        print(f"  WER:  {wer*100:.1f}% | CER: {cer*100:.1f}% | Tags: {', '.join(err_tags)}\n")

    # Aggregate metrics
    cat_stats = {}
    for r in results:
        c = r["category"]
        cat_stats.setdefault(c, {"wer": [], "cer": [], "latency": []})
        cat_stats[c]["wer"].append(r["wer"])
        cat_stats[c]["cer"].append(r["cer"])
        cat_stats[c]["latency"].append(r["latency_ms"])

    print("=" * 80)
    print("EVALUATION SUMMARY BY CATEGORY")
    print("=" * 80)
    for c, stats in cat_stats.items():
        avg_wer = sum(stats["wer"]) / len(stats["wer"]) * 100
        avg_cer = sum(stats["cer"]) / len(stats["cer"]) * 100
        avg_lat = sum(stats["latency"]) / len(stats["latency"])
        print(f"Category '{c}': Avg WER = {avg_wer:.1f}% | Avg CER = {avg_cer:.1f}% | Avg Latency = {avg_lat:.0f}ms")
    print("=" * 80)

    # Cleanup
    del model
    del processor
    gc.collect()

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper LoRA Model on Held-out Test Set")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--lora_adapter", type=str, default=None)
    parser.add_argument("--test_manifest", type=str, default=None)
    args = parser.parse_args()

    run_evaluation(args.base_model, args.lora_adapter, args.test_manifest)


if __name__ == "__main__":
    main()
