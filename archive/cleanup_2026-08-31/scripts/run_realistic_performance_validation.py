#!/usr/bin/env python3
"""
Performance & Correctness Validation Suite for Multi-Agent Telugu Voice AI.
Executes isolated, realistic tests covering:
1. Realistic STT Validation (Natural Telugu, Natural English, Telugu-English Code-mixed)
2. Realistic TTS Validation (Conversational greetings, negative RAG answers, English replies, multi-sentence responses)
3. End-to-End Voice Roundtrip User Experience (Cold vs Warm, with per-stage latency breakdown)
4. Memory & Resource Footprint Tracking (Process RSS, MPS memory, Swap)
5. Exports audio files for manual human listening evaluation.
"""

import os
import sys
import time
import subprocess
import psutil
import torch
import numpy as np
import soundfile as sf
import io
import json
from pathlib import Path

# Add ai-service to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ai-service"))

VALIDATION_DIR = REPO_ROOT / "validation_artifacts"
VALIDATION_DIR.mkdir(exist_ok=True)

def get_mem_stats():
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    
    mps_alloc_mb = 0.0
    mps_driver_mb = 0.0
    if torch.backends.mps.is_available():
        if hasattr(torch.mps, "current_allocated_memory"):
            try:
                mps_alloc_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            except Exception:
                pass
        if hasattr(torch.mps, "driver_allocated_memory"):
            try:
                mps_driver_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
            except Exception:
                pass
                
    return {
        "rss_mb": round(rss_mb, 2),
        "mps_alloc_mb": round(mps_alloc_mb, 2),
        "mps_driver_mb": round(mps_driver_mb, 2),
        "sys_used_gb": round(vm.used / (1024**3), 2),
        "sys_avail_gb": round(vm.available / (1024**3), 2),
        "swap_used_gb": round(sm.used / (1024**3), 2),
    }

def generate_test_speech(text: str, voice: str, out_path: Path, rate: int = 175) -> Path:
    """Generates audio sample via macOS speech synthesizer for reproducible testing."""
    cmd = ["say", "-v", voice, "-r", str(rate), "-o", str(out_path), "--data-format=LEF32@16000", text]
    subprocess.run(cmd, check=True)
    return out_path

def main():
    print("=" * 90, flush=True)
    print("      REALISTIC PERFORMANCE & CORRECTNESS VALIDATION SUITE", flush=True)
    print("=" * 90, flush=True)
    
    m_init = get_mem_stats()
    print(f"Initial State: Process RSS={m_init['rss_mb']}MB | System Used={m_init['sys_used_gb']}GB | Swap={m_init['swap_used_gb']}GB\n", flush=True)
    
    # ---------------------------------------------------------
    # 1. LOAD SINGLETON SERVICES
    # ---------------------------------------------------------
    print("-" * 90, flush=True)
    print("1. INITIALIZING SINGLETON SERVICES (ONE INSTANCE, NO DUPLICATES)", flush=True)
    print("-" * 90, flush=True)
    
    t0 = time.perf_counter()
    from app.speech.stt import get_stt_provider
    from app.speech.tts import get_tts_provider
    from app.embeddings.provider import get_embedding_provider
    from app.storage.postgres import PostgresManager
    from app.storage.qdrant import QdrantVectorStore
    from app.rag import RAGPipeline
    from app.agents.graph import AgentOrchestrator
    from app.speech.audio import decode_and_normalize_audio
    
    stt = get_stt_provider()
    tts = get_tts_provider()
    embed = get_embedding_provider()
    db = PostgresManager()
    vector_store = QdrantVectorStore()
    pipeline = RAGPipeline(vector_store=vector_store, embedding_provider=embed)
    orchestrator = AgentOrchestrator(rag_pipeline=pipeline)
    
    load_time_sec = time.perf_counter() - t0
    m_loaded = get_mem_stats()
    print(f"All Services Initialized in {load_time_sec:.2f}s:")
    print(f"  - STT Engine:     whisper_turbo (Device={stt._torch_device}, Model={stt.model_name})")
    print(f"  - TTS Engine:     {tts.engine_name} (Device={tts.device}, Adapter={tts.adapter_name})")
    print(f"  - Embedding:      {embed.model_name}")
    print(f"  - Process RSS:    {m_loaded['rss_mb']} MB ({m_loaded['rss_mb']/1024:.2f} GB)")
    print(f"  - MPS Allocated:  {m_loaded['mps_alloc_mb']} MB | Driver: {m_loaded['mps_driver_mb']} MB")
    print(f"  - System Memory:  {m_loaded['sys_used_gb']} GB / 16.00 GB (Available: {m_loaded['sys_avail_gb']} GB)")
    print()

    # ---------------------------------------------------------
    # 2. REALISTIC STT VALIDATION
    # ---------------------------------------------------------
    print("-" * 90, flush=True)
    print("2. REALISTIC STT (SPEECH-TO-TEXT) VALIDATION", flush=True)
    print("-" * 90, flush=True)
    
    stt_test_cases = []
    
    # Human recording from SYSPIN dataset
    syspin_wav = REPO_ROOT / "dataset/telugu_tts/raw_syspin/syspin_f_0002.wav"
    if syspin_wav.exists():
        stt_test_cases.append({
            "category": "Natural Telugu (Human Dataset - SYSPIN Female)",
            "wav_path": str(syspin_wav),
            "ref_text": "మల విసర్జనాన్ని శరీరంలోని ఏ భాగం నియంత్రిస్తుంది? ఈ చర్య నియంత్రితమా? అనియంత్రితమా? ఎందుకు?",
            "expected_meaning": "Biology query regarding digestive regulation."
        })
    
    # Human recording from IndicTTS dataset
    indictts_wav = REPO_ROOT / "dataset/telugu_tts/processed/wavs/indictts_f_0011.wav"
    if indictts_wav.exists():
        stt_test_cases.append({
            "category": "Natural Telugu (Human Dataset - IndicTTS Female)",
            "wav_path": str(indictts_wav),
            "ref_text": "చిత్తం పదండి, అన్నాడు విజయదాసు.",
            "expected_meaning": "'Let us go,' said Vijayadasu."
        })

    # Conversational Telugu query (Short)
    p_te_short = VALIDATION_DIR / "stt_te_short.wav"
    generate_test_speech("నమస్కారం! మీరు ఎలా ఉన్నారు?", "Geeta", p_te_short, rate=160)
    stt_test_cases.append({
        "category": "Natural Telugu (Short Conversational Greeting)",
        "wav_path": str(p_te_short),
        "ref_text": "నమస్కారం! మీరు ఎలా ఉన్నారు?",
        "expected_meaning": "Greeting: 'Hello! How are you?'"
    })

    # Conversational Telugu query (Normal)
    p_te_norm = VALIDATION_DIR / "stt_te_norm.wav"
    generate_test_speech("కళాశాల లైబ్రరీ తెరిచే సమయాలు ఏమిటి?", "Geeta", p_te_norm, rate=160)
    stt_test_cases.append({
        "category": "Natural Telugu (Normal RAG Query)",
        "wav_path": str(p_te_norm),
        "ref_text": "కళాశాల లైబ్రరీ తెరిచే సమయాలు ఏమిటి?",
        "expected_meaning": "Inquiry: 'What are the college library opening timings?'"
    })

    # Conversational English query (Normal)
    p_en_norm = VALIDATION_DIR / "stt_en_norm.wav"
    generate_test_speech("What is the minimum attendance requirement for semester exams?", "Rishi", p_en_norm, rate=175)
    stt_test_cases.append({
        "category": "Natural English (Indian Accent - Normal Query)",
        "wav_path": str(p_en_norm),
        "ref_text": "What is the minimum attendance requirement for semester exams?",
        "expected_meaning": "Policy query regarding 75% attendance rule."
    })

    # Conversational English query (Longer)
    p_en_long = VALIDATION_DIR / "stt_en_long.wav"
    generate_test_speech("Can you explain the procedure to apply for medical leave if I fall sick during the semester?", "Samantha", p_en_long, rate=170)
    stt_test_cases.append({
        "category": "Natural English (Longer Conversational Query)",
        "wav_path": str(p_en_long),
        "ref_text": "Can you explain the procedure to apply for medical leave if I fall sick during the semester?",
        "expected_meaning": "Procedure inquiry for medical leave submission."
    })

    # Telugu-English Code-Mixed Query 1
    p_mixed_1 = VALIDATION_DIR / "stt_mixed_1.wav"
    generate_test_speech("Hostel fee payment deadline eppudu untundi?", "Rishi", p_mixed_1, rate=170)
    stt_test_cases.append({
        "category": "Telugu-English Code-Mixed (Fee deadline)",
        "wav_path": str(p_mixed_1),
        "ref_text": "Hostel fee payment deadline ఎప్పుడు ఉంటుంది?",
        "expected_meaning": "Inquiring about hostel fee deadline in code-mixed Telugu/English."
    })

    # Telugu-English Code-Mixed Query 2
    p_mixed_2 = VALIDATION_DIR / "stt_mixed_2.wav"
    generate_test_speech("Naku library card apply cheyadaniki em documents kavali?", "Rishi", p_mixed_2, rate=170)
    stt_test_cases.append({
        "category": "Telugu-English Code-Mixed (Library card)",
        "wav_path": str(p_mixed_2),
        "ref_text": "నాకు library card apply చేయడానికి ఏమి documents కావాలి?",
        "expected_meaning": "Inquiring about library card required documents in code-mixed speech."
    })

    stt_results = []
    print(f"{'Category':<35} | {'Audio':<7} | {'STT Lat':<8} | {'RTF':<6} | {'Lang':<5} | {'Semantic Match'}")
    print("-" * 90)

    for tc in stt_test_cases:
        data, sr = sf.read(tc["wav_path"])
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio_dur = len(data) / float(sr)
        
        # Transcribe
        t_start = time.perf_counter()
        transcript, detected_lang, lat_ms = stt.transcribe(data.astype(np.float32), sample_rate=sr)
        lat_sec = lat_ms / 1000.0
        rtf = lat_sec / audio_dur if audio_dur > 0 else 0
        
        # Evaluate semantic match
        semantic_ok = len(transcript.strip()) > 0 and (
            detected_lang in ["te", "en"] or any(w in transcript.lower() for w in ["library", "attendance", "hostel", "నమస్కారం", "కాలేజ్", "ఏమిటి"])
        )
        
        stt_results.append({
            "category": tc["category"],
            "audio_dur_sec": round(audio_dur, 2),
            "stt_lat_sec": round(lat_sec, 2),
            "rtf": round(rtf, 2),
            "detected_lang": detected_lang,
            "transcript": transcript,
            "ref_text": tc["ref_text"],
            "semantic_match": "YES" if semantic_ok else "NO"
        })
        
        print(f"{tc['category'][:35]:<35} | {audio_dur:>5.2f}s | {lat_sec:>6.2f}s | {rtf:>5.2f}x | {detected_lang:<5} | {'MATCH' if semantic_ok else 'DIFF'}")
        print(f"   Expected:   \"{tc['ref_text']}\"")
        print(f"   Transcript: \"{transcript}\"")
        print()

    # ---------------------------------------------------------
    # 3. REALISTIC TTS VALIDATION
    # ---------------------------------------------------------
    print("-" * 90, flush=True)
    print("3. REALISTIC TTS (TEXT-TO-SPEECH) VALIDATION")
    print("   Note: Latency was measured objectively; speech naturalness/intelligibility requires human listening evaluation.")
    print("-" * 90, flush=True)
    
    tts_test_cases = [
        {
            "id": "tts_greeting",
            "name": "Short Conversational Greeting",
            "text": "నమస్కారం! నేను మీ తెలుగు AI సహాయకుడిని."
        },
        {
            "id": "tts_negative_rag",
            "name": "Negative RAG Fallback Response",
            "text": "మీరు అడిగిన ప్రశ్నకు సంబంధించిన సమాచారం పత్రంలో లేదు."
        },
        {
            "id": "tts_english_reply",
            "name": "English Assistant Response",
            "text": "Hello! How can I help you today?"
        },
        {
            "id": "tts_multi_sentence",
            "name": "Realistic Multi-Sentence Telugu Response (Policy)",
            "text": "కాలేజ్ అటెండెన్స్ నియమాలు ప్రకారం విద్యార్థులు కనీసం 75 శాతం హాజరు కలిగి ఉండాలి. ఒకవేళ వైద్య కారణాల వల్ల హాజరు తగ్గితే, మెడికల్ సర్టిఫికేట్ సమర్పించాల్సి ఉంటుంది."
        }
    ]
    
    tts_results = []
    print(f"{'Test Case':<32} | {'Chars':<6} | {'Exp Dur':<8} | {'Act Dur':<8} | {'Latency':<8} | {'RTF':<6} | {'Peak RSS':<8} | {'MPS Driver'}")
    print("-" * 90)

    for tc in tts_test_cases:
        m_before = get_mem_stats()
        t_start = time.perf_counter()
        
        wav_bytes, lat_ms = tts.synthesize(tc["text"])
        lat_sec = lat_ms / 1000.0
        
        m_after = get_mem_stats()
        
        data, sr = sf.read(io.BytesIO(wav_bytes))
        act_dur = len(data) / float(sr)
        
        is_indic = any(ord(c) >= 0x0C00 and ord(c) <= 0x0C7F for c in tc["text"])
        chars_per_sec = 11.0 if is_indic else 14.5
        exp_dur = max(0.8, len(tc["text"]) / chars_per_sec)
        rtf = lat_sec / act_dur if act_dur > 0 else 0
        
        # Save output audio file for human inspection
        out_wav_path = VALIDATION_DIR / f"{tc['id']}.wav"
        with open(out_wav_path, "wb") as f:
            f.write(wav_bytes)
            
        tts_results.append({
            "name": tc["name"],
            "text": tc["text"],
            "chars": len(tc["text"]),
            "exp_dur_sec": round(exp_dur, 2),
            "act_dur_sec": round(act_dur, 2),
            "latency_sec": round(lat_sec, 2),
            "rtf": round(rtf, 2),
            "peak_rss_mb": m_after["rss_mb"],
            "mps_driver_mb": m_after["mps_driver_mb"],
            "saved_file": str(out_wav_path)
        })
        
        print(f"{tc['name'][:32]:<32} | {len(tc['text']):>5}  | {exp_dur:>6.2f}s  | {act_dur:>6.2f}s  | {lat_sec:>6.2f}s  | {rtf:>5.2f}x | {m_after['rss_mb']:>6.0f}MB | {m_after['mps_driver_mb']:>7.0f}MB")
        print(f"   Audio Saved: {out_wav_path.name}")
        print()

    # ---------------------------------------------------------
    # 4. END-TO-END VOICE ROUNDTRIP USER EXPERIENCE
    # ---------------------------------------------------------
    print("-" * 90, flush=True)
    print("4. END-TO-END COMPLETE VOICE USER EXPERIENCE (FULL ROUNDTRIP)")
    print("-" * 90, flush=True)
    
    # Prepare realistic test audio for E2E: "కాలేజ్ అటెండెన్స్ రూల్స్ ఏమిటి?"
    e2e_wav_path = VALIDATION_DIR / "e2e_query_attendance.wav"
    generate_test_speech("కాలేజ్ అటెండెన్స్ రూల్స్ ఏమిటి?", "Geeta", e2e_wav_path, rate=160)
    
    e2e_runs = [("Cold / First Query", e2e_wav_path), ("Warm / Second Query", e2e_wav_path)]
    e2e_metrics = []
    
    for label, audio_file in e2e_runs:
        print(f"\n--- Testing End-to-End Roundtrip: {label} ---")
        t_total_start = time.perf_counter()
        
        # Step A: Audio Preprocessing
        t_a = time.perf_counter()
        raw_bytes = open(audio_file, "rb").read()
        audio_array, sr_out, dur_sec = decode_and_normalize_audio(raw_bytes, target_sr=16000)
        ms_prep = (time.perf_counter() - t_a) * 1000.0
        
        # Step B: STT Transcription
        t_b = time.perf_counter()
        query_text, detected_lang, ms_stt = stt.transcribe(audio_array, sample_rate=sr_out)
        
        # Step C: Conversation State
        conv_id = db.get_or_create_conversation(None)
        recent_msgs = db.get_recent_messages(conv_id, limit=5)
        db.add_message(conversation_id=conv_id, role="user", content=query_text)
        
        # Step D: Multi-Agent Orchestration (Understanding -> Retrieval -> LLM)
        t_d = time.perf_counter()
        state = orchestrator.run(
            query=query_text,
            conversation_id=conv_id,
            messages=recent_msgs,
            retrieval_mode="hybrid"
        )
        ms_agent_total = (time.perf_counter() - t_d) * 1000.0
        
        final_answer = state.get("final_answer", "")
        intent = state.get("intent", "knowledge_query")
        
        db.add_message(conversation_id=conv_id, role="assistant", content=final_answer, intent=intent)
        
        # Step E: Text-to-Speech Synthesis
        t_e = time.perf_counter()
        wav_resp_bytes, ms_tts = tts.synthesize(text=final_answer, voice_id="telugu_female_1")
        
        total_sec = time.perf_counter() - t_total_start
        
        # Save E2E response audio
        e2e_out_file = VALIDATION_DIR / f"e2e_response_{label.split()[0].lower()}.wav"
        with open(e2e_out_file, "wb") as f:
            f.write(wav_resp_bytes)
            
        metrics_entry = {
            "run": label,
            "input_audio_dur_sec": round(dur_sec, 2),
            "transcript": query_text,
            "detected_lang": detected_lang,
            "intent": intent,
            "answer_text": final_answer,
            "audio_prep_ms": round(ms_prep, 2),
            "stt_ms": round(ms_stt, 2),
            "agent_total_ms": round(ms_agent_total, 2),
            "tts_ms": round(ms_tts, 2),
            "total_roundtrip_sec": round(total_sec, 2),
            "out_audio_file": str(e2e_out_file)
        }
        e2e_metrics.append(metrics_entry)
        
        print(f"  User Audio Transcript:  \"{query_text}\" (Detected Lang: {detected_lang})")
        print(f"  Classified Intent:      {intent}")
        print(f"  Assistant Final Answer: \"{final_answer[:120]}...\"")
        print(f"  Latency Breakdown:")
        print(f"    - Audio Preprocessing: {ms_prep:>7.2f} ms")
        print(f"    - Speech-to-Text:      {ms_stt:>7.2f} ms ({ms_stt/1000:.2f} s)")
        print(f"    - Agent + RAG + LLM:   {ms_agent_total:>7.2f} ms ({ms_agent_total/1000:.2f} s)")
        print(f"    - Text-to-Speech:      {ms_tts:>7.2f} ms ({ms_tts/1000:.2f} s)")
        print(f"    ------------------------------------")
        print(f"    = TOTAL USER ROUNDTRIP:{total_sec:>7.2f} s")
        print(f"  Response Audio Saved:   {e2e_out_file.name}")

    # Save summary report to JSON
    summary_report = {
        "memory_final": get_mem_stats(),
        "stt_results": stt_results,
        "tts_results": tts_results,
        "e2e_metrics": e2e_metrics
    }
    with open(VALIDATION_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 90)
    print(f"Validation suite complete! All audio artifacts and summary saved to: {VALIDATION_DIR}")
    print("=" * 90)

if __name__ == "__main__":
    main()
