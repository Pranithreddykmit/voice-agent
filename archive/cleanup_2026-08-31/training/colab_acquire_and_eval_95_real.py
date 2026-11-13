# ==============================================================================
# COLAB REAL-AUDIO ACQUISITION & 95-SAMPLE BASELINE EVALUATION PIPELINE
# ==============================================================================
# Model: openai/whisper-large-v3-turbo (Production Configuration)
# Manifest: validation_artifacts/asr_real_eval_manifest.json (95 Real Human Samples)
#
# STRICT FORENSIC PROTOCOL:
# - ZERO SIMULATION.
# - ZERO SYNTHETIC SPEECH AS GROUND TRUTH.
# - Gated datasets without credentials are marked UNAVAILABLE (no fabrication).
# - Explicitly distinguishes:
#   1. Actual Recognition Errors
#   2. Telugu <-> Latin Representation Differences
#   3. Script Bleed (Devanagari / Tamil)
#   4. Hallucinations
# ==============================================================================

import os
import sys
import time
import gc
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torchaudio
import soundfile as sf
import numpy as np

# Verify GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Device: {device} | PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"[INIT] Active GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)")

# Directory structure
AUDIO_OUT_DIR = Path("/content/asr_95_audio") if os.path.exists("/content") else Path("./asr_95_audio")
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON_PATH = Path("/content/whisper_turbo_95_real_baseline.json") if os.path.exists("/content") else Path("validation_artifacts/whisper_turbo_95_real_baseline.json")

# ==============================================================================
# DOMAIN ENTITY LIST & CODE-MIXED TRANSLITERATION DICTIONARY
# ==============================================================================
DOMAIN_ENTITIES = [
    "attendance", "semester", "exams", "hostel", "fee", "library",
    "medical leave", "dean", "application", "deadline", "qdrant",
    "postgresql", "rag", "embedding", "sgpa", "cgpa", "revaluation",
    "hall ticket", "condonation", "malpractice", "wifi", "password",
    "credentials", "placement", "credits", "curriculum"
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
    "password": ["పాస్వర్డ్"],
    "placement": ["ప్లేస్‌మెంట్"],
    "credits": ["క్రెడిట్లు"],
    "curriculum": ["కరికులమ్"],
    "qdrant": ["క్యూడ్రాంట్"],
    "postgresql": ["పోస్ట్‌గ్రేస్"],
    "rag": ["రాగ్"],
    "embedding": ["ఎంబెడ్డింగ్"]
}

# ==============================================================================
# LEVENSHTEIN METRIC FUNCTIONS
# ==============================================================================
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

def normalize_representation(text: str) -> str:
    """Normalizes code-mixed Telugu-English synonyms to English Latin base for fair evaluation."""
    norm = text.lower()
    for eng_term, syn_list in ENTITY_SYNONYMS.items():
        for syn in syn_list:
            norm = norm.replace(syn, eng_term)
    return norm

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

def classify_error_type(ref: str, hyp: str, raw_wer: float, norm_wer: float, bleed: Dict[str, Any]) -> str:
    """Classifies prediction error into exact forensic categories."""
    if raw_wer == 0.0:
        return "NONE (PERFECT_MATCH)"
    if len(hyp) == 0 or len(hyp) > 3.5 * len(ref):
        return "HALLUCINATION"
    if bleed["has_bleed"]:
        return "SCRIPT_BLEED"
    if raw_wer > 0.3 and norm_wer <= 0.15:
        return "SCRIPT_REPRESENTATION_DIFFERENCE"
    return "ACTUAL_RECOGNITION_ERROR"

# ==============================================================================
# PRODUCTION WHISPER TURBO LOADER
# ==============================================================================
class ProductionWhisperTurbo:
    def __init__(self, device="cuda"):
        self.device = device
        self.name = "openai/whisper-large-v3-turbo"
        print(f"\n[WhisperTurbo] Loading {self.name} on {self.device}...")
        t0 = time.perf_counter()
        
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        self.processor = AutoProcessor.from_pretrained(self.name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        
        # Bilingual domain prompt conditioning
        prompt_text = "attendance, semester exams, hostel, library, fee, medical leave, హాజరు, పరీక్షలు, ఫీజు, దరఖాస్తు."
        raw_prompt = self.processor.get_prompt_ids(prompt_text)
        self.prompt_ids = torch.tensor(raw_prompt, device=self.device)
        self.init_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[WhisperTurbo] Initialized in {self.init_ms:.2f} ms")

    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=dtype)
        
        with torch.no_grad():
            out = self.model.generate(
                input_features,
                task="transcribe",
                prompt_ids=self.prompt_ids,
                max_new_tokens=256
            )
            raw_text = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
        if self.device == "cuda":
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        
        # Repetition filter
        collapsed = re.sub(r"(.{1,6}?)\1{3,}", r"\1", raw_text).strip()
        collapsed = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", collapsed, flags=re.IGNORECASE).strip()
        return collapsed, lat_ms

# ==============================================================================
# MAIN ACQUISITION & EVALUATION HARNESS
# ==============================================================================
def execute_95_real_audio_baseline(manifest_path: str):
    print("=" * 80)
    print("EXECUTING FULL 95-SAMPLE REAL-AUDIO BASELINE EVALUATION")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    samples = manifest_data["samples"]
    print(f"[MANIFEST] Loaded {len(samples)} sample specifications.")
    
    # Initialize Model
    stt = ProductionWhisperTurbo(device=device)
    
    predictions = []
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", os.environ.get("HF_TOKEN", None))
    
    # Dataset iterators cache for streaming
    dataset_streams = {}
    
    for idx, s in enumerate(samples, start=1):
        s_id = s["id"]
        source = s["dataset_source"]
        ref = s["ref_transcript"]
        dur_s = s.get("duration_sec", 4.5)
        category = s["category"]
        
        audio_array = None
        sr = s.get("sample_rate_hz", 16000)
        status_msg = "ACQUIRED"
        
        # ----------------------------------------------------------------------
        # Real Audio Acquisition Strategy
        # ----------------------------------------------------------------------
        try:
            if "google/fleurs" in source:
                split_name = "te_in" if "(te_in" in source else "en_us"
                if split_name not in dataset_streams:
                    from datasets import load_dataset
                    dataset_streams[split_name] = iter(load_dataset("google/fleurs", split_name, split="test", streaming=True))
                stream_item = next(dataset_streams[split_name])
                audio_array = np.array(stream_item["audio"]["array"], dtype=np.float32)
                sr = stream_item["audio"]["sampling_rate"]
                
            elif "librispeech_asr" in source:
                if "librispeech" not in dataset_streams:
                    from datasets import load_dataset
                    dataset_streams["librispeech"] = iter(load_dataset("librispeech_asr", "clean", split="test", streaming=True))
                stream_item = next(dataset_streams["librispeech"])
                audio_array = np.array(stream_item["audio"]["array"], dtype=np.float32)
                sr = stream_item["audio"]["sampling_rate"]
                
            elif "IndicVoices" in source or "Kathbath" in source:
                if not hf_token:
                    status_msg = "UNAVAILABLE (Requires HF Token for Gated Access)"
                else:
                    repo_id = "ai4bharat/IndicVoices" if "IndicVoices" in source else "ai4bharat/IndicSuperb"
                    sub_name = "te" if "IndicVoices" in source else "kathbath_te"
                    if sub_name not in dataset_streams:
                        from datasets import load_dataset
                        dataset_streams[sub_name] = iter(load_dataset(repo_id, sub_name, split="test", streaming=True, token=hf_token))
                    stream_item = next(dataset_streams[sub_name])
                    audio_array = np.array(stream_item["audio"]["array"], dtype=np.float32)
                    sr = stream_item["audio"]["sampling_rate"]
                    
            elif "CoSTA" in source:
                # Load or clone CoSTA test WAV
                costa_local_wav = AUDIO_OUT_DIR / f"{s_id}.wav"
                if costa_local_wav.exists():
                    audio_array, sr = sf.read(str(costa_local_wav))
                else:
                    # In online Colab environment, fetch directly from CSALT repo
                    status_msg = "UNAVAILABLE (CoSTA test audio requires GitHub repo clone)"
        except Exception as acq_err:
            status_msg = f"UNAVAILABLE ({str(acq_err)[:60]})"
            
        # ----------------------------------------------------------------------
        # Model Inference Execution (Strictly on real audio)
        # ----------------------------------------------------------------------
        if audio_array is not None:
            # Resample to 16kHz if necessary
            if sr != 16000:
                import scipy.signal, math
                gcd = math.gcd(int(sr), 16000)
                audio_array = scipy.signal.resample_poly(audio_array, 16000 // gcd, int(sr) // gcd).astype(np.float32)
                sr = 16000
                
            hyp, lat_ms = stt.transcribe(audio_array, sample_rate=sr)
            dur_calc = len(audio_array) / float(sr)
            rtf = (lat_ms / 1000.0) / dur_calc if dur_calc > 0 else 0.0
            
            raw_cer = compute_cer(ref, hyp)
            raw_wer = compute_wer(ref, hyp)
            
            norm_ref = normalize_representation(ref)
            norm_hyp = normalize_representation(hyp)
            norm_wer = compute_wer(norm_ref, norm_hyp)
            
            bleed = compute_script_bleed(hyp)
            en_pres = check_english_preservation(ref, hyp)
            tel_corr = check_telugu_script_correctness(hyp)
            entity_info = check_entity_preservation(ref, hyp)
            err_class = classify_error_type(ref, hyp, raw_wer, norm_wer, bleed)
            
            pred_record = {
                "sample_id": s_id,
                "status": "EVALUATED",
                "category": category,
                "language": s.get("language", "te"),
                "dataset_source": source,
                "speaker_id": s.get("speaker_id", "unknown"),
                "duration_sec": round(dur_calc, 2),
                "ref_transcript": ref,
                "hyp_transcript": hyp,
                "latency_ms": round(lat_ms, 2),
                "rtf": round(rtf, 4),
                "raw_cer": round(raw_cer, 4),
                "raw_wer": round(raw_wer, 4),
                "norm_wer": round(norm_wer, 4),
                "english_token_preservation": en_pres,
                "telugu_script_correctness": tel_corr,
                "script_bleed": bleed,
                "hallucination": (err_class == "HALLUCINATION"),
                "entity_preservation": entity_info,
                "error_classification": err_class
            }
            predictions.append(pred_record)
            print(f"[{idx:02d}/95] {s_id:<15} | {category:<28} | CER: {raw_cer:.3f} | WER: {raw_wer:.3f} | Class: {err_class}")
        else:
            # Strictly record UNAVAILABLE entry without simulation
            pred_record = {
                "sample_id": s_id,
                "status": status_msg,
                "category": category,
                "language": s.get("language", "te"),
                "dataset_source": source,
                "speaker_id": s.get("speaker_id", "unknown"),
                "duration_sec": dur_s,
                "ref_transcript": ref,
                "hyp_transcript": None,
                "latency_ms": None,
                "rtf": None,
                "raw_cer": None,
                "raw_wer": None,
                "norm_wer": None,
                "english_token_preservation": None,
                "telugu_script_correctness": None,
                "script_bleed": None,
                "hallucination": False,
                "entity_preservation": None,
                "error_classification": "DATA_UNAVAILABLE"
            }
            predictions.append(pred_record)
            print(f"[{idx:02d}/95] {s_id:<15} | {category:<28} | Status: {status_msg}")

    # --------------------------------------------------------------------------
    # Aggregate Evaluation Summary
    # --------------------------------------------------------------------------
    evaluated_preds = [p for p in predictions if p["status"] == "EVALUATED"]
    unavailable_preds = [p for p in predictions if p["status"] != "EVALUATED"]
    
    summary = {
        "manifest_total_entries": len(predictions),
        "successfully_evaluated_count": len(evaluated_preds),
        "unavailable_count": len(unavailable_preds),
        "overall_evaluated_metrics": {
            "mean_cer": round(float(np.mean([p["raw_cer"] for p in evaluated_preds])), 4) if evaluated_preds else None,
            "mean_raw_wer": round(float(np.mean([p["raw_wer"] for p in evaluated_preds])), 4) if evaluated_preds else None,
            "mean_norm_wer": round(float(np.mean([p["norm_wer"] for p in evaluated_preds])), 4) if evaluated_preds else None,
            "mean_english_preservation": round(float(np.mean([p["english_token_preservation"] for p in evaluated_preds])), 4) if evaluated_preds else None,
            "mean_telugu_script_correctness": round(float(np.mean([p["telugu_script_correctness"] for p in evaluated_preds])), 4) if evaluated_preds else None,
            "mean_entity_recall": round(float(np.mean([p["entity_preservation"]["entity_recall"] for p in evaluated_preds])), 4) if evaluated_preds else None,
            "mean_latency_ms": round(float(np.mean([p["latency_ms"] for p in evaluated_preds])), 2) if evaluated_preds else None,
            "mean_rtf": round(float(np.mean([p["rtf"] for p in evaluated_preds])), 4) if evaluated_preds else None,
            "error_distribution": {
                "perfect_match_count": sum(1 for p in evaluated_preds if p["error_classification"] == "NONE (PERFECT_MATCH)"),
                "actual_recognition_errors": sum(1 for p in evaluated_preds if p["error_classification"] == "ACTUAL_RECOGNITION_ERROR"),
                "script_representation_differences": sum(1 for p in evaluated_preds if p["error_classification"] == "SCRIPT_REPRESENTATION_DIFFERENCE"),
                "script_bleed_count": sum(1 for p in evaluated_preds if p["error_classification"] == "SCRIPT_BLEED"),
                "hallucination_count": sum(1 for p in evaluated_preds if p["error_classification"] == "HALLUCINATION")
            }
        }
    }

    final_deliverable = {
        "metadata": {
            "model_name": "openai/whisper-large-v3-turbo",
            "device": device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "execution_protocol": "GENUINE_NEURAL_FORWARD_PASS_NO_SIMULATION"
        },
        "summary": summary,
        "predictions": predictions
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_deliverable, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 80)
    print("95-SAMPLE REAL BASELINE RUN COMPLETED")
    print(f"Output saved to: {RESULTS_JSON_PATH}")
    print("=" * 80)
    return final_deliverable

if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else "validation_artifacts/asr_real_eval_manifest.json"
    execute_95_real_audio_baseline(manifest)
