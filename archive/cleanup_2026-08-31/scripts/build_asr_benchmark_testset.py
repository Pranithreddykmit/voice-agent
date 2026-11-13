"""
Benchmark Test Set Builder for Telugu/English/Code-Mixed ASR Investigation.
Creates and indexes 14 realistic audio samples across 6 categories:
A: Pure Telugu (including Human recordings)
B: Pure English (Indian accent)
C: Telugu-English Code-Mixed (Highest Priority)
D: Conversational Telugu with Technical English Terms
E: Short Questions
F: Longer Conversational Inquiries
"""

import os
import subprocess
from pathlib import Path
import soundfile as sf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "validation_artifacts" / "benchmark_test_set"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_speech(text: str, voice: str, out_path: Path, rate: int = 165):
    """Generates 16kHz mono WAV using macOS speech synthesis."""
    tmp_path = out_path.with_suffix(".aiff")
    cmd = ["say", "-v", voice, "-r", str(rate), "-o", str(tmp_path), text]
    subprocess.run(cmd, check=True)
    
    # Convert and normalize to 16kHz mono float32 WAV
    data, sr = sf.read(str(tmp_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        import math
        gcd = math.gcd(sr, 16000)
        up = 16000 // gcd
        down = sr // gcd
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        sr = 16000
    sf.write(str(out_path), data.astype(np.float32), 16000, subtype="PCM_16")
    if tmp_path.exists():
        tmp_path.unlink()

TEST_SAMPLES = [
    # --- Category A: Pure Telugu ---
    {
        "id": "A1_te_human_syspin",
        "category": "A. Pure Telugu",
        "source": "Human (SYSPIN Dataset)",
        "wav_path": str(REPO_ROOT / "dataset/telugu_tts/raw_syspin/syspin_f_0002.wav"),
        "ref_text": "మల విసర్జనాన్ని శరీరంలోని ఏ భాగం నియంత్రిస్తుంది? ఈ చర్య నియంత్రితమా? అనియంత్రితమా? ఎందుకు?",
        "meaning": "Biology query on digestive regulation."
    },
    {
        "id": "A2_te_human_indictts",
        "category": "A. Pure Telugu",
        "source": "Human (IndicTTS Dataset)",
        "wav_path": str(REPO_ROOT / "dataset/telugu_tts/processed/wavs/indictts_f_0011.wav"),
        "ref_text": "చిత్తం పదండి, అన్నాడు విజయదాసు.",
        "meaning": "Story narrative: 'Let us go,' said Vijayadasu."
    },
    {
        "id": "A3_te_synthetic",
        "category": "A. Pure Telugu",
        "source": "Synthetic (Geeta te_IN)",
        "gen_text": "విశ్వవిద్యాలయ గ్రంథాలయం ప్రతిరోజూ ఉదయం ఎనిమిది గంటల నుండి సాయంత్రం ఎనిమిది గంటల వరకు తెరిచి ఉంటుంది.",
        "voice": "Geeta",
        "ref_text": "విశ్వవిద్యాలయ గ్రంథాలయం ప్రతిరోజూ ఉదయం ఎనిమిది గంటల నుండి సాయంత్రం ఎనిమిది గంటల వరకు తెరిచి ఉంటుంది.",
        "meaning": "University library is open daily from 8 AM to 8 PM."
    },

    # --- Category B: Pure English ---
    {
        "id": "B1_en_policy",
        "category": "B. Pure English",
        "source": "Synthetic (Rishi en_IN)",
        "gen_text": "What is the minimum attendance requirement for semester exams?",
        "voice": "Rishi",
        "ref_text": "What is the minimum attendance requirement for semester exams?",
        "meaning": "Inquiry about 75% attendance policy."
    },
    {
        "id": "B2_en_hostel",
        "category": "B. Pure English",
        "source": "Synthetic (Tara en_IN)",
        "gen_text": "How can students apply for hostel accommodation in the second year?",
        "voice": "Tara",
        "ref_text": "How can students apply for hostel accommodation in the second year?",
        "meaning": "Hostel room allocation question."
    },

    # --- Category C: Telugu-English Code-Mixed (HIGHEST PRIORITY) ---
    {
        "id": "C1_mixed_attendance",
        "category": "C. Telugu-English Code-Mixed",
        "source": "Synthetic (Rishi en_IN)",
        "gen_text": "Semester exams ki minimum attendance entha undali?",
        "voice": "Rishi",
        "ref_text": "Semester exams ki minimum attendance entha undali?",
        "meaning": "Code-mixed student query: What minimum attendance is required for semester exams?"
    },
    {
        "id": "C2_mixed_fee_deadline",
        "category": "C. Telugu-English Code-Mixed",
        "source": "Synthetic (Rishi en_IN)",
        "gen_text": "Hostel fee payment deadline eppudu untundi?",
        "voice": "Rishi",
        "ref_text": "Hostel fee payment deadline eppudu untundi?",
        "meaning": "Code-mixed student query: When is the hostel fee payment deadline?"
    },
    {
        "id": "C3_mixed_library_docs",
        "category": "C. Telugu-English Code-Mixed",
        "source": "Synthetic (Rishi en_IN)",
        "gen_text": "Naku library card apply cheyadaniki em documents kavali?",
        "voice": "Rishi",
        "ref_text": "Naku library card apply cheyadaniki em documents kavali?",
        "meaning": "Code-mixed student query: What documents do I need to apply for a library card?"
    },
    {
        "id": "C4_mixed_medical_leave",
        "category": "C. Telugu-English Code-Mixed",
        "source": "Synthetic (Rishi en_IN)",
        "gen_text": "Naku medical leave apply cheyyali, procedure enti?",
        "voice": "Rishi",
        "ref_text": "Naku medical leave apply cheyyali, procedure enti?",
        "meaning": "Code-mixed student query: I need to apply for medical leave, what is the procedure?"
    },

    # --- Category D: Conversational Telugu with English Technical Terms ---
    {
        "id": "D1_te_tech_terms",
        "category": "D. Conversational Telugu + Technical Terms",
        "source": "Synthetic (Geeta te_IN)",
        "gen_text": "కళాశాల లైబ్రరీ తెరిచే సమయాలు ఏమిటి?",
        "voice": "Geeta",
        "ref_text": "కళాశాల లైబ్రరీ తెరిచే సమయాలు ఏమిటి?",
        "meaning": "Telugu inquiry with English loanword 'Library'."
    },
    {
        "id": "D2_mixed_wifi_login",
        "category": "D. Conversational Telugu + Technical Terms",
        "source": "Synthetic (Geeta te_IN)",
        "gen_text": "కాలేజీ క్యాంపస్ లో వైఫై పాస్వర్డ్ మార్చడం ఎలా?",
        "voice": "Geeta",
        "ref_text": "కాలేజీ క్యాంపస్ లో వైఫై పాస్వర్డ్ మార్చడం ఎలా?",
        "meaning": "Technical query with loanwords: College, Campus, Wi-Fi, Password."
    },

    # --- Category E: Short Questions ---
    {
        "id": "E1_te_short_greeting",
        "category": "E. Short Questions",
        "source": "Synthetic (Geeta te_IN)",
        "gen_text": "నమస్కారం! మీరు ఎలా ఉన్నారు?",
        "voice": "Geeta",
        "ref_text": "నమస్కారం! మీరు ఎలా ఉన్నారు?",
        "meaning": "Short greeting: Hello! How are you?"
    },
    {
        "id": "E2_mixed_exam_deadline",
        "category": "E. Short Questions",
        "source": "Synthetic (Rishi en_IN)",
        "gen_text": "Exam fee deadline eppudu?",
        "voice": "Rishi",
        "ref_text": "Exam fee deadline eppudu?",
        "meaning": "Short question: When is the exam fee deadline?"
    },

    # --- Category F: Longer Conversational Questions ---
    {
        "id": "F1_en_long_medical",
        "category": "F. Longer Conversational Questions",
        "source": "Synthetic (Samantha en_US)",
        "gen_text": "Can you explain the procedure to apply for medical leave if I fall sick during the semester?",
        "voice": "Samantha",
        "ref_text": "Can you explain the procedure to apply for medical leave if I fall sick during the semester?",
        "meaning": "Long multi-clause English inquiry."
    },
    {
        "id": "F2_mixed_long_reschedule",
        "category": "F. Longer Conversational Questions",
        "source": "Synthetic (Rishi en_IN)",
        "gen_text": "Mid term exams reschedule chesukovadaniki Dean permission teesukovalandi, application form ekkada submit cheyyali?",
        "voice": "Rishi",
        "ref_text": "Mid term exams reschedule chesukovadaniki Dean permission teesukovalandi, application form ekkada submit cheyyali?",
        "meaning": "Long complex code-mixed question with multiple academic constraints."
    }
]

def build_test_set():
    manifest = []
    for item in TEST_SAMPLES:
        sample_id = item["id"]
        out_wav = OUT_DIR / f"{sample_id}.wav"
        
        if "wav_path" in item and Path(item["wav_path"]).exists():
            # Copy or convert human audio to 16kHz mono WAV
            data, sr = sf.read(item["wav_path"])
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                import scipy.signal
                import math
                gcd = math.gcd(sr, 16000)
                up = 16000 // gcd
                down = sr // gcd
                data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
                sr = 16000
            sf.write(str(out_wav), data.astype(np.float32), 16000, subtype="PCM_16")
        else:
            generate_speech(item["gen_text"], item["voice"], out_wav)
            
        data, sr = sf.read(str(out_wav))
        duration = len(data) / float(sr)
        
        entry = {
            "id": sample_id,
            "category": item["category"],
            "source": item["source"],
            "wav_path": str(out_wav),
            "duration_sec": round(duration, 2),
            "ref_text": item["ref_text"],
            "meaning": item["meaning"]
        }
        manifest.append(entry)
        print(f"Prepared [{sample_id}] ({duration:.2f}s) - {item['category']}")
        
    import json
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nSuccessfully generated {len(manifest)} benchmark samples at {OUT_DIR}")

if __name__ == "__main__":
    build_test_set()
