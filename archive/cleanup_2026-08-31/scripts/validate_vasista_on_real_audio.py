"""
Independent Validation Engine for vasista22/whisper-telugu-large-v2 on Real Audio
==================================================================================
Performs empirical, uncompromised validation of vasista22/whisper-telugu-large-v2:
1. Loads the actual pretrained model weights from Hugging Face Hub
2. Runs live forward-pass inference on 35 distinct REAL audio files:
   - 10 Human Telugu from SYSPIN
   - 10 Human Telugu from IndicTTS
   - 10 Telugu-English Code-Mixed
   - 5 English Academic Inquiries
3. Calculates exact Word Error Rate (WER), Character Error Rate (CER),
   Devanagari/Tamil script bleed, and English regression.
4. Outputs raw machine-readable audit data to validation_artifacts/vasista_real_audio_validation.json
"""

import os
import gc
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import soundfile as sf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = REPO_ROOT / "validation_artifacts" / "vasista_real_audio_validation.json"


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


def is_romanized(ref: str, hyp: str) -> bool:
    has_telugu_ref = any("\u0C00" <= c <= "\u0C7F" for c in ref)
    is_pure_latin_hyp = bool(re.match(r"^[a-zA-Z0-9\s.,?!'-]+$", hyp.strip())) if hyp.strip() else False
    return has_telugu_ref and is_pure_latin_hyp


def build_35_sample_real_eval_manifest() -> List[Dict]:
    """Assembles 35 real audio files from SYSPIN, IndicTTS, code-mixed, and English sets."""
    samples = []

    # 1. Real Human SYSPIN Telugu (10 samples)
    with open(REPO_ROOT / "dataset" / "telugu_tts" / "syspin_selected_manifest.json") as f:
        syspin_data = json.load(f)
    for idx, item in enumerate(syspin_data[5:15]):
        wav_path = REPO_ROOT / "dataset" / "telugu_tts" / "raw_syspin" / f"{item['id']}.wav"
        if wav_path.exists():
            samples.append({
                "id": item["id"],
                "source": "Human (IISc SYSPIN Studio)",
                "category": "pure_telugu_human_syspin",
                "wav_path": str(wav_path),
                "ref_text": item["text"],
                "duration_sec": item.get("duration_sec", 6.0)
            })

    # 2. Real Human IndicTTS Telugu (10 samples)
    with open(REPO_ROOT / "dataset" / "telugu_tts" / "indictts_selected_manifest.json") as f:
        indictts_data = json.load(f)
    for idx, item in enumerate(indictts_data[5:15]):
        wav_path = REPO_ROOT / "dataset" / "telugu_tts" / "raw_indictts" / f"{item['id']}.wav"
        if wav_path.exists():
            samples.append({
                "id": item["id"],
                "source": "Human (IIT Madras IndicTTS Studio)",
                "category": "pure_telugu_human_indictts",
                "wav_path": str(wav_path),
                "ref_text": item["text"],
                "duration_sec": item.get("duration_sec", 5.0)
            })

    # 3. Telugu-English Code-Mixed Queries (10 samples)
    cm_items = [
        {"id": "cm_val_01", "wav": "C1_mixed_attendance.wav", "text": "Semester exams ki minimum attendance entha undali?", "dur": 3.22},
        {"id": "cm_val_02", "wav": "C2_mixed_fee_deadline.wav", "text": "Hostel fee payment deadline eppudu untundi?", "dur": 2.65},
        {"id": "cm_val_03", "wav": "C3_mixed_library_docs.wav", "text": "Naku library card apply cheyadaniki em documents kavali?", "dur": 3.88},
        {"id": "cm_val_04", "wav": "C4_mixed_medical_leave.wav", "text": "Naku medical leave apply cheyyali, procedure enti?", "dur": 3.46},
        {"id": "cm_val_05", "wav": "D2_mixed_wifi_login.wav", "text": "కాలేజీ క్యాంపస్ లో వైఫై పాస్వర్డ్ మార్చడం ఎలా?", "dur": 3.31},
        {"id": "cm_val_06", "wav": "E2_mixed_exam_deadline.wav", "text": "Exam fee deadline eppudu?", "dur": 1.66},
        {"id": "cm_val_07", "wav": "F2_mixed_long_reschedule.wav", "text": "Mid term exams reschedule chesukovadaniki Dean permission teesukovalandi, application form ekkada submit cheyyali?", "dur": 7.2},
        {"id": "cm_val_08", "wav": "stt_mixed_1.wav", "text": "Semester exams fee payment last date eppudu?", "dur": 3.6},
        {"id": "cm_val_09", "wav": "stt_mixed_2.wav", "text": "Library card apply cheyadaniki fee entha pay cheyyali?", "dur": 4.1},
        {"id": "cm_val_10", "wav": "e2e_query_attendance.wav", "text": "Attendance 75% lekapothe semester exams rayavacha?", "dur": 3.8}
    ]
    for c in cm_items:
        wav_p = REPO_ROOT / "validation_artifacts" / "benchmark_test_set" / c["wav"]
        if not wav_p.exists():
            wav_p = REPO_ROOT / "validation_artifacts" / c["wav"]
        if wav_p.exists():
            samples.append({
                "id": c["id"],
                "source": "Telugu-English Code-Mixed Spoken Audio",
                "category": "code_mixed_telugu_english",
                "wav_path": str(wav_p),
                "ref_text": c["text"],
                "duration_sec": c["dur"]
            })

    # 4. Pure English Queries (5 samples)
    en_items = [
        {"id": "en_val_01", "wav": "B1_en_policy.wav", "text": "What is the minimum attendance requirement for semester exams?", "dur": 3.89},
        {"id": "en_val_02", "wav": "B2_en_hostel.wav", "text": "How can students apply for hostel accommodation in the second year?", "dur": 3.91},
        {"id": "en_val_03", "wav": "F1_en_long_medical.wav", "text": "Can you explain the procedure to apply for medical leave if I fall sick during the semester?", "dur": 4.82},
        {"id": "en_val_04", "wav": "stt_en_norm.wav", "text": "What is the procedure to apply for hostel accommodation?", "dur": 4.1},
        {"id": "en_val_05", "wav": "stt_en_long.wav", "text": "Can you explain the detailed guidelines for examination revaluation and recounting?", "dur": 5.3}
    ]
    for e in en_items:
        wav_p = REPO_ROOT / "validation_artifacts" / "benchmark_test_set" / e["wav"]
        if not wav_p.exists():
            wav_p = REPO_ROOT / "validation_artifacts" / e["wav"]
        if wav_p.exists():
            samples.append({
                "id": e["id"],
                "source": "Academic English Spoken Audio",
                "category": "pure_english",
                "wav_path": str(wav_p),
                "ref_text": e["text"],
                "duration_sec": e["dur"]
            })

    return samples


def run_vasista_live_evaluation():
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    samples = build_35_sample_real_eval_manifest()
    print("=" * 80)
    print(f"RUNNING INDEPENDENT LIVE VALIDATION FOR vasista22/whisper-telugu-large-v2")
    print(f"Real Audio Samples to Evaluate: {len(samples)}")
    print("=" * 80)

    model_id = "vasista22/whisper-telugu-large-v2"
    device = "cpu"
    dtype = torch.float32

    print(f"Loading processor and model '{model_id}'...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # Get Telugu decoder prompt tokens
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="te", task="transcribe")

    eval_results = []
    category_metrics = {}

    for s in samples:
        audio_path = s["wav_path"]
        ref_text = s["ref_text"]
        cat = s["category"]

        audio_data, sr = sf.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)

        # Resample to 16kHz if needed
        if sr != 16000:
            import scipy.signal
            import math
            gcd = math.gcd(int(sr), 16000)
            audio_data = scipy.signal.resample_poly(audio_data, 16000 // gcd, int(sr) // gcd).astype(np.float32)

        t0 = time.perf_counter()
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=256
            )

        hyp_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        wer = compute_wer(ref_text, hyp_text)
        cer = compute_cer(ref_text, hyp_text)
        is_dev = has_devanagari(hyp_text) and not has_devanagari(ref_text)
        is_tam = has_tamil(hyp_text) and not has_tamil(ref_text)
        is_rom = is_romanized(ref_text, hyp_text)

        category_metrics.setdefault(cat, {"wers": [], "cers": [], "latencies": []})
        category_metrics[cat]["wers"].append(wer)
        category_metrics[cat]["cers"].append(cer)
        category_metrics[cat]["latencies"].append(elapsed_ms)

        entry = {
            "id": s["id"],
            "source": s["source"],
            "category": cat,
            "duration_sec": s["duration_sec"],
            "ref_text": ref_text,
            "vasista_prediction": hyp_text,
            "wer": wer,
            "cer": cer,
            "latency_ms": round(elapsed_ms, 1),
            "devanagari_bleed": is_dev,
            "tamil_bleed": is_tam,
            "romanized": is_rom
        }
        eval_results.append(entry)

        print(f"\n[{s['id']}] ({cat}) | Dur: {s['duration_sec']}s | Latency: {elapsed_ms:.0f}ms")
        print(f"  Ref:      {ref_text}")
        print(f"  Vasista:  {hyp_text}")
        print(f"  WER: {wer*100:.1f}% | CER: {cer*100:.1f}% | Dev: {is_dev} | Tam: {is_tam}")

    # Summary
    total_wers = [r["wer"] for r in eval_results]
    total_cers = [r["cer"] for r in eval_results]

    summary = {
        "model_id": model_id,
        "total_samples_evaluated": len(eval_results),
        "overall_wer_pct": round(sum(total_wers) / len(total_wers) * 100, 2),
        "overall_cer_pct": round(sum(total_cers) / len(total_cers) * 100, 2),
        "category_summary": {
            cat: {
                "count": len(d["wers"]),
                "avg_wer_pct": round(sum(d["wers"]) / len(d["wers"]) * 100, 2),
                "avg_cer_pct": round(sum(d["cers"]) / len(d["cers"]) * 100, 2),
                "avg_latency_ms": round(sum(d["latencies"]) / len(d["latencies"]), 1)
            }
            for cat, d in category_metrics.items()
        },
        "sample_evaluations": eval_results
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("INDEPENDENT LIVE EVALUATION SUMMARY FOR vasista22/whisper-telugu-large-v2")
    print("=" * 80)
    print(f"Total Samples Evaluated: {len(eval_results)}")
    print(f"REAL Overall WER:        {summary['overall_wer_pct']}%")
    print(f"REAL Overall CER:        {summary['overall_cer_pct']}%")
    for cat, d in summary["category_summary"].items():
        print(f"  • {cat}: WER = {d['avg_wer_pct']}% | CER = {d['avg_cer_pct']}%")
    print("=" * 80)

    # Cleanup memory
    del model
    del processor
    gc.collect()

    return summary


if __name__ == "__main__":
    run_vasista_live_evaluation()
