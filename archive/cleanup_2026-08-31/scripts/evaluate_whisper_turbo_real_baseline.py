"""
Empirical Real-Audio Baseline Evaluation for Production Whisper Large-v3-Turbo.
Executes genuine forward-pass neural inference on all physically available audio files.
Strictly NO SIMULATION. Measures CER, WER, Latency, RTF, Script Bleed, and Entity Retention.
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import soundfile as sf

# Set project roots
BASE_DIR = Path(__file__).resolve().parent.parent
AI_SERVICE_DIR = BASE_DIR / "ai-service"
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from app.speech.stt import get_stt_provider, WhisperSTTProvider
from app.speech.audio import decode_and_normalize_audio

# Domain Entities for RAG Retention Measurement
DOMAIN_ENTITIES = [
    "attendance", "semester", "exams", "hostel", "fee", "library",
    "medical leave", "dean", "application", "deadline", "qdrant",
    "postgresql", "rag", "embedding", "sgpa", "cgpa", "revaluation",
    "hall ticket", "condonation", "malpractice", "wifi", "password"
]

ENTITY_SYNONYMS = {
    "attendance": ["అటెండెన్స్", "హాజరు"],
    "semester": ["సెమిస్టర్"],
    "exams": ["ఎగ్జామ్స్", "పరీక్షలు"],
    "hostel": ["హాస్టల్"],
    "fee": ["ఫీజు"],
    "library": ["లైబ్రరీ", "గ్రంథాలయం"],
    "medical leave": ["మెడికల్ లీవ్", "వైద్య సెలవు"],
    "dean": ["డీన్"],
    "application": ["అప్లికేషన్", "దరఖాస్తు"],
    "deadline": ["డెడ్‌లైన్", "గడువు"],
    "revaluation": ["రీవాల్యుయేషన్"],
    "hall ticket": ["హాల్ టికెట్"],
    "condonation": ["కండోనేషన్"],
    "wifi": ["వైఫై"],
    "password": ["పాస్వర్డ్"]
}

def compute_wer(ref: str, hyp: str) -> float:
    r_words = re.findall(r"\w+", ref.lower())
    h_words = re.findall(r"\w+", hyp.lower())
    if not r_words:
        return 0.0 if not h_words else 1.0
    dp = [[0] * (len(h_words) + 1) for _ in range(len(r_words) + 1)]
    for i in range(len(r_words) + 1):
        dp[i][0] = i
    for j in range(len(h_words) + 1):
        dp[0][j] = j
    for i in range(1, len(r_words) + 1):
        for j in range(1, len(h_words) + 1):
            cost = 0 if r_words[i - 1] == h_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return min(1.0, dp[len(r_words)][len(h_words)] / float(len(r_words)))

def compute_cer(ref: str, hyp: str) -> float:
    r_clean = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", ref.lower())).strip()
    h_clean = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", hyp.lower())).strip()
    if not r_clean:
        return 0.0 if not h_clean else 1.0
    dp = [[0] * (len(h_clean) + 1) for _ in range(len(r_clean) + 1)]
    for i in range(len(r_clean) + 1):
        dp[i][0] = i
    for j in range(len(h_clean) + 1):
        dp[0][j] = j
    for i in range(1, len(r_clean) + 1):
        for j in range(1, len(h_clean) + 1):
            cost = 0 if r_clean[i - 1] == h_clean[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return min(1.0, dp[len(r_clean)][len(h_clean)] / float(len(r_clean)))

def compute_script_bleed(text: str) -> Dict[str, Any]:
    dev_matches = re.findall(r"[\u0900-\u097F]", text)
    tam_matches = re.findall(r"[\u0B80-\u0BFF]", text)
    return {
        "devanagari_count": len(dev_matches),
        "tamil_count": len(tam_matches),
        "has_bleed": (len(dev_matches) > 0 or len(tam_matches) > 0)
    }

def check_english_preservation(ref_text: str, hyp_text: str) -> float:
    ref_en = re.findall(r"\b[a-zA-Z]{2,}\b", ref_text.lower())
    if not ref_en:
        return 1.0
    hyp_lower = hyp_text.lower()
    preserved = sum(1 for token in ref_en if token in hyp_lower)
    return round(preserved / float(len(ref_en)), 4)

def check_telugu_script_correctness(hyp_text: str) -> float:
    tel_chars = len(re.findall(r"[\u0C00-\u0C7F]", hyp_text))
    foreign_indic = len(re.findall(r"[\u0900-\u097F\u0B80-\u0BFF]", hyp_text))
    total_indic = tel_chars + foreign_indic
    if total_indic == 0:
        return 1.0
    return round(tel_chars / float(total_indic), 4)

def check_entity_preservation(ref_text: str, hyp_text: str) -> Dict[str, Any]:
    ref_lower = ref_text.lower()
    hyp_lower = hyp_text.lower()
    present_in_ref = []
    preserved_in_hyp = []
    
    for entity in DOMAIN_ENTITIES:
        if entity in ref_lower:
            present_in_ref.append(entity)
            if entity in hyp_lower:
                preserved_in_hyp.append(entity)
            else:
                synonyms = ENTITY_SYNONYMS.get(entity, [])
                if any(syn in hyp_text for syn in synonyms):
                    preserved_in_hyp.append(entity)
                    
    recall = (len(preserved_in_hyp) / len(present_in_ref)) if present_in_ref else 1.0
    return {
        "entities_in_ref": present_in_ref,
        "entities_preserved": preserved_in_hyp,
        "entity_recall": round(recall, 4)
    }

def run_real_whisper_turbo_baseline():
    print("=" * 80)
    print("EXECUTING REAL AUDIO BASELINE: PRODUCTION WHISPER LARGE-V3-TURBO")
    print("=" * 80)
    
    # 1. Initialize Production Provider
    t0 = time.perf_counter()
    stt = get_stt_provider()
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[STT] Initialized {stt.model_name} on device '{stt._torch_device}' in {init_ms:.2f} ms")
    
    # 2. Collect Real Physical Audio Test Samples
    # A. 15 Production Quarantined Benchmark Samples
    benchmark_manifest_path = BASE_DIR / "validation_artifacts" / "benchmark_test_set" / "manifest.json"
    with open(benchmark_manifest_path, "r", encoding="utf-8") as f:
        benchmark_samples = json.load(f)
        
    eval_set = []
    for s in benchmark_samples:
        p = Path(s["wav_path"])
        if p.exists():
            eval_set.append({
                "id": s["id"],
                "category": s["category"],
                "source": s["source"],
                "wav_path": str(p),
                "ref_text": s["ref_text"],
                "duration_sec": s.get("duration_sec", 0.0)
            })
            
    # B. Additional Real Studio Samples from SYSPIN & IndicTTS
    syspin_manifest_path = BASE_DIR / "dataset" / "telugu_tts" / "syspin_selected_manifest.json"
    if syspin_manifest_path.exists():
        with open(syspin_manifest_path, "r", encoding="utf-8") as f:
            syspin_items = json.load(f)
            count = 0
            for item in syspin_items:
                if item["id"] != "syspin_f_0002" and count < 5:  # Avoid A1 leak
                    wav_file = BASE_DIR / "dataset" / "telugu_tts" / "raw_syspin" / f"{item['id']}.wav"
                    if wav_file.exists():
                        eval_set.append({
                            "id": f"studio_{item['id']}",
                            "category": "A. Pure Telugu (Studio Real)",
                            "source": "IISc SYSPIN Real Studio",
                            "wav_path": str(wav_file),
                            "ref_text": item["text"],
                            "duration_sec": item.get("duration", 0.0)
                        })
                        count += 1

    indictts_manifest_path = BASE_DIR / "dataset" / "telugu_tts" / "indictts_selected_manifest.json"
    if indictts_manifest_path.exists():
        with open(indictts_manifest_path, "r", encoding="utf-8") as f:
            indictts_items = json.load(f)
            count = 0
            for item in indictts_items:
                if item["id"] != "indictts_f_0011" and count < 5:  # Avoid A2 leak
                    wav_file = BASE_DIR / "dataset" / "telugu_tts" / "raw_indictts" / f"{item['id']}.wav"
                    if wav_file.exists():
                        eval_set.append({
                            "id": f"studio_{item['id']}",
                            "category": "A. Pure Telugu (Studio Real)",
                            "source": "IIT Madras IndicTTS Real Studio",
                            "wav_path": str(wav_file),
                            "ref_text": item["text"],
                            "duration_sec": item.get("duration", 0.0)
                        })
                        count += 1

    print(f"\n[EVAL] Collected {len(eval_set)} verified physical audio test samples.")
    
    predictions = []
    
    for idx, sample in enumerate(eval_set, start=1):
        wav_path = sample["wav_path"]
        with open(wav_path, "rb") as f:
            raw_bytes = f.read()
            
        audio_array, sr, dur = decode_and_normalize_audio(raw_bytes, target_sr=16000)
        
        # Genuine Neural Forward Pass
        hyp, lang, lat_ms = stt.transcribe(audio_array, sample_rate=sr)
        
        # Strict sanity check
        assert isinstance(hyp, str), "Hypothesis must be a valid string"
        
        ref = sample["ref_text"]
        cer = compute_cer(ref, hyp)
        wer = compute_wer(ref, hyp)
        
        bleed = compute_script_bleed(hyp)
        en_pres = check_english_preservation(ref, hyp)
        tel_corr = check_telugu_script_correctness(hyp)
        entity_res = check_entity_preservation(ref, hyp)
        
        dur_s = dur if dur > 0 else sample["duration_sec"]
        rtf = round((lat_ms / 1000.0) / dur_s, 4) if dur_s > 0 else 0.0
        
        pred_entry = {
            "id": sample["id"],
            "category": sample["category"],
            "source": sample["source"],
            "duration_sec": round(dur_s, 2),
            "ref_text": ref,
            "hyp_text": hyp,
            "detected_lang": lang,
            "latency_ms": round(lat_ms, 2),
            "rtf": rtf,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "english_token_preservation": en_pres,
            "telugu_script_correctness": tel_corr,
            "script_bleed": bleed,
            "hallucination": bool(len(hyp) > 3 * len(ref) or len(hyp) == 0),
            "entity_preservation": entity_res
        }
        predictions.append(pred_entry)
        print(f"[{idx:02d}/{len(eval_set):02d}] {sample['id']:<26} | CER: {cer:.3f} | WER: {wer:.3f} | Lat: {lat_ms:.1f}ms | Hyp: {hyp[:60]}")

    # Summary by Category
    categories = sorted(list(set(p["category"] for p in predictions)))
    category_metrics = {}
    for cat in categories:
        cat_preds = [p for p in predictions if p["category"] == cat]
        category_metrics[cat] = {
            "count": len(cat_preds),
            "mean_cer": round(float(np.mean([p["cer"] for p in cat_preds])), 4),
            "mean_wer": round(float(np.mean([p["wer"] for p in cat_preds])), 4),
            "mean_latency_ms": round(float(np.mean([p["latency_ms"] for p in cat_preds])), 2),
            "mean_rtf": round(float(np.mean([p["rtf"] for p in cat_preds])), 4),
            "mean_english_preservation": round(float(np.mean([p["english_token_preservation"] for p in cat_preds])), 4),
            "mean_entity_recall": round(float(np.mean([p["entity_preservation"]["entity_recall"] for p in cat_preds])), 4),
            "bleed_count": sum(1 for p in cat_preds if p["script_bleed"]["has_bleed"])
        }

    overall_metrics = {
        "total_samples": len(predictions),
        "mean_cer": round(float(np.mean([p["cer"] for p in predictions])), 4),
        "mean_wer": round(float(np.mean([p["wer"] for p in predictions])), 4),
        "mean_latency_ms": round(float(np.mean([p["latency_ms"] for p in predictions])), 2),
        "mean_rtf": round(float(np.mean([p["rtf"] for p in predictions])), 4),
        "mean_english_preservation": round(float(np.mean([p["english_token_preservation"] for p in predictions])), 4),
        "mean_telugu_script_correctness": round(float(np.mean([p["telugu_script_correctness"] for p in predictions])), 4),
        "mean_entity_recall": round(float(np.mean([p["entity_preservation"]["entity_recall"] for p in predictions])), 4),
        "total_bleed_samples": sum(1 for p in predictions if p["script_bleed"]["has_bleed"]),
        "total_hallucinations": sum(1 for p in predictions if p["hallucination"])
    }

    report = {
        "metadata": {
            "model_name": stt.model_name,
            "torch_device": stt._torch_device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prompt_conditioning": True,
            "execution_type": "GENUINE_NEURAL_FORWARD_PASS"
        },
        "overall_summary": overall_metrics,
        "category_summary": category_metrics,
        "predictions": predictions
    }

    out_json = BASE_DIR / "validation_artifacts" / "whisper_turbo_real_baseline.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("REAL ASR BASELINE COMPLETE")
    print(f"Results saved to: {out_json}")
    print("=" * 80)
    return report

if __name__ == "__main__":
    run_real_whisper_turbo_baseline()
