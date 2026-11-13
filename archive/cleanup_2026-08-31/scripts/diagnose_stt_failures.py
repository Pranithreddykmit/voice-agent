"""
Forensic STT Diagnostic & Language Detection Analysis.
Investigates the root causes of Whisper STT language hallucination, audio pipeline integrity,
and compares 5 decoding configurations across 10 representative voice recordings:

Conditions:
A. Current Production Configuration (Language auto-detection + _ACADEMIC_PROMPT)
B. Explicit Telugu Language (language='te')
C. Explicit English Language (language='en')
D. Language Auto-Detection + No Prompt (baseline unconditioned)
E. Language Auto-Detection + Current Academic Prompt (explicit check)

Analyzes audio physics: format, channels, sample rate, duration, peak/RMS energy,
Whisper language identification probabilities, raw tokens, and sanitized transcripts.
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
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Project roots
BASE_DIR = Path(__file__).resolve().parent.parent
AI_SERVICE_DIR = BASE_DIR / "ai-service"
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from app.speech.audio import decode_and_normalize_audio, is_silence
from app.speech.stt import _ACADEMIC_PROMPT

def get_audio_signal_metrics(raw_bytes: bytes, audio_array: np.ndarray, orig_sr: int, orig_channels: int) -> Dict[str, Any]:
    """Measures physical and digital signal health metrics."""
    peak = float(np.max(np.abs(audio_array))) if len(audio_array) > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio_array ** 2))) if len(audio_array) > 0 else 0.0
    dbfs = 20 * np.log10(rms) if rms > 1e-7 else -100.0
    clipping_samples = int(np.sum(np.abs(audio_array) >= 0.999))
    
    return {
        "raw_byte_size": len(raw_bytes),
        "orig_sample_rate": orig_sr,
        "orig_channels": orig_channels,
        "preprocessed_sr": 16000,
        "preprocessed_channels": 1,
        "duration_sec": round(len(audio_array) / 16000.0, 3),
        "peak_amplitude": round(peak, 4),
        "rms_amplitude": round(rms, 4),
        "dbfs_level": round(dbfs, 2),
        "clipping_sample_count": clipping_samples,
        "is_silent": is_silence(audio_array)
    }

def detect_whisper_language_logits(model, processor, input_features, device) -> Dict[str, float]:
    """Extracts top language probabilities from Whisper encoder output."""
    with torch.no_grad():
        encoder = model.get_encoder()
        encoder_outputs = encoder(input_features)
        
        # In Whisper, decoder starts with <|startoftranscript|>
        bos_token_id = processor.tokenizer.bos_token_id or model.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[bos_token_id]], device=device)
        
        logits = model(
            input_features=None,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs
        ).logits[0, -1]
        
        # Extract lang_to_id mapping
        lang_to_id = getattr(model.generation_config, "lang_to_id", None) or getattr(model.config, "lang_to_id", {})
        if not lang_to_id:
            # Fallback: scan tokenizer vocab for <|lang|> tokens
            lang_to_id = {}
            for token, tid in processor.tokenizer.get_vocab().items():
                if token.startswith("<|") and token.endswith("|>") and len(token) <= 6:
                    code = token.strip("<|>")
                    if code not in ["startoftranscript", "endoftext", "transcribe", "translate", "notimestamps"]:
                        lang_to_id[code] = tid
                        
        lang_probs = {}
        for lang_code, token_id in lang_to_id.items():
            if token_id < logits.shape[0]:
                lang_probs[lang_code] = float(logits[token_id].item())
                
        # Softmax over language tokens
        if lang_probs:
            max_val = max(lang_probs.values())
            exp_vals = {k: np.exp(v - max_val) for k, v in lang_probs.items()}
            sum_exp = sum(exp_vals.values())
            normalized_probs = {k: round(float(v / sum_exp), 4) for k, v in exp_vals.items()}
            top_sorted = dict(sorted(normalized_probs.items(), key=lambda item: item[1], reverse=True)[:8])
            return top_sorted
        return {"unknown": 1.0}

def run_stt_forensic_suite():
    print("=" * 80)
    print("EXECUTING STT FORENSIC DIAGNOSTIC & LANGUAGE DETECTION AUDIT")
    print("=" * 80)
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")
    
    model_name = "openai/whisper-large-v3-turbo"
    print(f"[INIT] Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    # Prompt IDs
    raw_prompt = processor.get_prompt_ids(_ACADEMIC_PROMPT)
    prompt_ids = torch.tensor(raw_prompt, device=device)
    
    # Define 10 Controlled Test Audio Recordings (5 pure Telugu, 5 Telugu-English code-mixed)
    test_recordings = [
        # 5 Pure Telugu
        {
            "id": "diag_te_01_syspin_real",
            "category": "Pure Telugu",
            "type": "Real Studio Human",
            "wav_path": str(BASE_DIR / "dataset" / "telugu_tts" / "raw_syspin" / "syspin_f_0003.wav"),
            "ref_transcript": "స్టోలన్లు, వాలిస్నేరియా, స్ట్రాబేరీ, లసునాలు, ఉల్లి, కొమ్ము..."
        },
        {
            "id": "diag_te_02_indictts_real",
            "category": "Pure Telugu",
            "type": "Real Studio Human",
            "wav_path": str(BASE_DIR / "dataset" / "telugu_tts" / "raw_indictts" / "indictts_f_0003.wav"),
            "ref_transcript": "ఆవుతో అడావుడి."
        },
        {
            "id": "diag_te_03_syspin_admissions",
            "category": "Pure Telugu",
            "type": "Real Studio Human",
            "wav_path": str(BASE_DIR / "dataset" / "telugu_tts" / "raw_syspin" / "syspin_f_0006.wav"),
            "ref_transcript": "అయితే, అమెరికా ప్రభుత్వం కారణంగా, హితేశ్ పోటీ చేయడానికి సాంకేతిక కారణాలు ఎదురయ్యాయి."
        },
        {
            "id": "diag_te_04_greeting_norm",
            "category": "Pure Telugu",
            "type": "Recorded Voice",
            "wav_path": str(BASE_DIR / "validation_artifacts" / "stt_te_norm.wav"),
            "ref_transcript": "నమస్కారం! మీరు ఎలా ఉన్నారు?"
        },
        {
            "id": "diag_te_05_greeting_short",
            "category": "Pure Telugu",
            "type": "Recorded Voice",
            "wav_path": str(BASE_DIR / "validation_artifacts" / "stt_te_short.wav"),
            "ref_transcript": "నమస్కారం, సహాయం కావాలి."
        },
        
        # 5 Telugu-English Code-Mixed
        {
            "id": "diag_mixed_01_attendance",
            "category": "Code-Mixed",
            "type": "Recorded Voice",
            "wav_path": str(BASE_DIR / "validation_artifacts" / "e2e_query_attendance.wav"),
            "ref_transcript": "Semester exams ki minimum attendance entha undali?"
        },
        {
            "id": "diag_mixed_02_medical_leave",
            "category": "Code-Mixed",
            "type": "Recorded Voice",
            "wav_path": str(BASE_DIR / "validation_artifacts" / "benchmark_test_set" / "C4_mixed_medical_leave.wav"),
            "ref_transcript": "Naku medical leave apply cheyyali, procedure enti?"
        },
        {
            "id": "diag_mixed_03_library_card",
            "category": "Code-Mixed",
            "type": "Recorded Voice",
            "wav_path": str(BASE_DIR / "validation_artifacts" / "benchmark_test_set" / "C3_mixed_library_docs.wav"),
            "ref_transcript": "Naku library card apply cheyadaniki em documents kavali?"
        },
        {
            "id": "diag_mixed_04_wifi_it",
            "category": "Code-Mixed",
            "type": "Recorded Voice",
            "wav_path": str(BASE_DIR / "validation_artifacts" / "benchmark_test_set" / "D2_mixed_wifi_login.wav"),
            "ref_transcript": "కాలేజీ క్యాంపస్ లో వైఫై పాస్వర్డ్ మార్చడం ఎలా?"
        },
        {
            "id": "diag_mixed_05_revaluation",
            "category": "Code-Mixed",
            "type": "Recorded Voice",
            "wav_path": str(BASE_DIR / "validation_artifacts" / "stt_mixed_2.wav"),
            "ref_transcript": "Exam fee deadline eppudu untundi?"
        }
    ]
    
    diagnostic_results = []
    
    for idx, item in enumerate(test_recordings, start=1):
        wav_p = Path(item["wav_path"])
        if not wav_p.exists():
            print(f"[{idx}/10] WARNING: File {wav_p} not found. Skipping.")
            continue
            
        with open(wav_p, "rb") as f:
            raw_bytes = f.read()
            
        # Get raw container metadata
        data_orig, orig_sr = sf.read(wav_p)
        orig_channels = data_orig.shape[1] if data_orig.ndim > 1 else 1
        
        # Audio preprocessing
        audio_array, sr, dur = decode_and_normalize_audio(raw_bytes, target_sr=16000)
        signal_metrics = get_audio_signal_metrics(raw_bytes, audio_array, orig_sr, orig_channels)
        
        # Feature extraction
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        # Extract Language Logits
        lang_probabilities = detect_whisper_language_logits(model, processor, input_features, device)
        top_lang = list(lang_probabilities.keys())[0] if lang_probabilities else "unknown"
        top_prob = list(lang_probabilities.values())[0] if lang_probabilities else 0.0
        
        # Condition A: Current Production (Auto-Detect + Academic Prompt)
        with torch.no_grad():
            out_A = model.generate(input_features, task="transcribe", prompt_ids=prompt_ids, max_new_tokens=256)
            hyp_A = processor.batch_decode(out_A, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
        # Condition B: Explicit Telugu (language="te", task="transcribe")
        with torch.no_grad():
            out_B = model.generate(input_features, language="te", task="transcribe", max_new_tokens=256)
            hyp_B = processor.batch_decode(out_B, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
        # Condition C: Explicit English (language="en", task="transcribe")
        with torch.no_grad():
            out_C = model.generate(input_features, language="en", task="transcribe", max_new_tokens=256)
            hyp_C = processor.batch_decode(out_C, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
        # Condition D: Auto-Detect + NO Prompt
        with torch.no_grad():
            out_D = model.generate(input_features, task="transcribe", max_new_tokens=256)
            hyp_D = processor.batch_decode(out_D, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
        # Condition E: Auto-Detect + Current Academic Prompt (explicit test)
        with torch.no_grad():
            out_E = model.generate(input_features, task="transcribe", prompt_ids=prompt_ids, max_new_tokens=256)
            hyp_E = processor.batch_decode(out_E, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
        record = {
            "id": item["id"],
            "category": item["category"],
            "type": item["type"],
            "wav_path": str(wav_p),
            "ref_transcript": item["ref_transcript"],
            "signal_metrics": signal_metrics,
            "detected_language_probabilities": lang_probabilities,
            "predicted_top_language": top_lang,
            "predicted_top_language_prob": top_prob,
            "decoding_conditions": {
                "condition_A_production_auto_with_prompt": hyp_A,
                "condition_B_explicit_telugu": hyp_B,
                "condition_C_explicit_english": hyp_C,
                "condition_D_auto_no_prompt": hyp_D,
                "condition_E_auto_with_academic_prompt": hyp_E
            }
        }
        diagnostic_results.append(record)
        print(f"\n[{idx:02d}/10] {item['id']:<28}")
        print(f"      Signal: {signal_metrics['duration_sec']}s, dBFS: {signal_metrics['dbfs_level']} dB, Clip: {signal_metrics['clipping_sample_count']}")
        print(f"      Top Lang: {top_lang} ({top_prob*100:.1f}%) | All: {lang_probabilities}")
        print(f"      Hyp A (Prod Auto+Prompt) : {hyp_A}")
        print(f"      Hyp B (Forced Telugu)   : {hyp_B}")
        print(f"      Hyp C (Forced English)  : {hyp_C}")
        print(f"      Hyp D (Auto No Prompt)  : {hyp_D}")

    report = {
        "metadata": {
            "title": "Forensic STT Diagnostic & Language Detection Analysis",
            "model_name": model_name,
            "device": device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_diagnosed_recordings": len(diagnostic_results)
        },
        "diagnostic_evaluations": diagnostic_results
    }
    
    out_json = BASE_DIR / "validation_artifacts" / "STT_FAILURE_FORENSIC_ANALYSIS.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 80)
    print("STT FORENSIC DIAGNOSTIC COMPLETE")
    print(f"Results saved to: {out_json}")
    print("=" * 80)
    return report

if __name__ == "__main__":
    run_stt_forensic_suite()
