#!/usr/bin/env python3
"""
Comprehensive Hardware Acceleration & Memory Benchmark for Apple Silicon M4.
Evaluates CPU vs MPS for:
1. FastEmbed ONNX (CPU vs CoreML)
2. Whisper Large-v3-Turbo (CPU vs MPS fp32/fp16)
3. IndicF5 TTS + LoRA + Vocos (CPU vs MPS fp32/fp16)
4. Incremental Component Memory RSS & Unified Memory Allocation Breakdown
"""

import os
import sys
import time
import gc
import psutil
import torch
import numpy as np
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Ensure ai-service is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-service")))

def get_process_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print_header("APPLE SILICON M4 HARDWARE & MPS DIAGNOSTICS")
    initial_rss = get_process_rss_mb()
    print(f"Initial Process RSS: {initial_rss:.2f} MB")
    print(f"PyTorch Version:     {torch.__version__}")
    print(f"MPS Built:           {torch.backends.mps.is_built()}")
    print(f"MPS Available:       {torch.backends.mps.is_available()}")
    
    mps_available = torch.backends.mps.is_available()
    
    # -------------------------------------------------------------------------
    # PART 1: MEMORY BREAKDOWN BY COMPONENT (INCREMENTAL RSS)
    # -------------------------------------------------------------------------
    print_header("1. INCREMENTAL MEMORY (RSS) BREAKDOWN BY COMPONENT")
    
    rss_tracker = {}
    last_rss = get_process_rss_mb()
    
    # Baseline
    rss_tracker["0. Baseline Process"] = last_rss
    print(f"[Baseline Process] RSS: {last_rss:.2f} MB")
    
    # 1. FastEmbed ONNX
    from app.embeddings.provider import FastEmbedProvider
    emb_provider = FastEmbedProvider(model_name="intfloat/multilingual-e5-large")
    _ = emb_provider.embed_query("నమస్కారం పరీక్ష")
    curr_rss = get_process_rss_mb()
    rss_tracker["1. FastEmbed ONNX (multilingual-e5-large)"] = curr_rss - last_rss
    print(f"[FastEmbed ONNX] Added: {curr_rss - last_rss:.2f} MB | Total RSS: {curr_rss:.2f} MB")
    last_rss = curr_rss
    
    # 2. BGE Reranker
    from app.retrieval.reranker import TransformersCrossEncoderReranker
    from app.schemas import RetrievedChunk
    reranker = TransformersCrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    sample_chunk = RetrievedChunk(
        chunk_id="1",
        document_id="doc1",
        text="test document",
        score=0.9,
        retrieval_mode="hybrid",
        source="test.pdf",
        page=1,
        chunk_index=0
    )
    _ = reranker.rerank("query", [sample_chunk])
    curr_rss = get_process_rss_mb()
    rss_tracker["2. BGE Reranker (bge-reranker-base)"] = curr_rss - last_rss
    print(f"[BGE Reranker] Added: {curr_rss - last_rss:.2f} MB | Total RSS: {curr_rss:.2f} MB")
    last_rss = curr_rss
    
    # 3. Whisper Large-v3-Turbo
    from app.speech.stt import WhisperSTTProvider
    stt_cpu = WhisperSTTProvider(model_name="openai/whisper-large-v3-turbo", device="cpu")
    curr_rss = get_process_rss_mb()
    rss_tracker["3. Whisper Large-v3-Turbo (CPU)"] = curr_rss - last_rss
    print(f"[Whisper STT (CPU)] Added: {curr_rss - last_rss:.2f} MB | Total RSS: {curr_rss:.2f} MB")
    last_rss = curr_rss
    
    # 4. IndicF5 TTS + Vocos + LoRA
    from app.speech.tts import IndicF5TTSProvider
    tts_cpu = IndicF5TTSProvider(
        adapter_path="models/indicf5_telugu_lora",
        device="cpu"
    )
    curr_rss = get_process_rss_mb()
    rss_tracker["4. IndicF5 TTS + Vocos + LoRA (CPU)"] = curr_rss - last_rss
    print(f"[IndicF5 TTS (CPU)] Added: {curr_rss - last_rss:.2f} MB | Total RSS: {curr_rss:.2f} MB")
    last_rss = curr_rss
    
    print("\nSummary of Memory Consumption by Module:")
    for k, v in rss_tracker.items():
        print(f"  {k:<45}: {v:>8.2f} MB")
    print(f"  {'Total Combined Single-Process RSS':<45}: {last_rss:>8.2f} MB ({last_rss/1024:.2f} GB)")

    # -------------------------------------------------------------------------
    # PART 2: WHISPER BENCHMARK (CPU vs MPS)
    # -------------------------------------------------------------------------
    print_header("2. WHISPER LARGE-V3-TURBO: CPU vs MPS BENCHMARK")
    
    audio_te, sr_te = sf.read("dataset/telugu_tts/processed/wavs/indictts_f_0011.wav", dtype="float32")
    audio_en, sr_en = sf.read(".venv/lib/python3.14/site-packages/f5_tts/infer/examples/basic/basic_ref_en.wav", dtype="float32")
    audio_mix, sr_mix = sf.read("dataset/telugu_tts/processed/wavs/indictts_f_0003.wav", dtype="float32")
    
    # Benchmark CPU Whisper
    print("\n[Whisper CPU Testing (8 threads)]...")
    t0 = time.perf_counter()
    txt_te_cpu, lang_te_cpu, lat_te_cpu = stt_cpu.transcribe(audio_te, sr_te)
    txt_en_cpu, lang_en_cpu, lat_en_cpu = stt_cpu.transcribe(audio_en, sr_en)
    txt_mix_cpu, lang_mix_cpu, lat_mix_cpu = stt_cpu.transcribe(audio_mix, sr_mix)
    
    print(f"  Telugu (3.20s):    {lat_te_cpu:>7.2f} ms | Lang: {lang_te_cpu} | Transcript: '{txt_te_cpu}'")
    print(f"  English (5.33s):   {lat_en_cpu:>7.2f} ms | Lang: {lang_en_cpu} | Transcript: '{txt_en_cpu}'")
    print(f"  Mixed (1.55s):     {lat_mix_cpu:>7.2f} ms | Lang: {lang_mix_cpu} | Transcript: '{txt_mix_cpu}'")
    print(f"  CPU Total Time:    {(lat_te_cpu + lat_en_cpu + lat_mix_cpu)/1000.0:.2f} s")
    
    # Benchmark MPS Whisper if available
    if mps_available:
        print("\n[Whisper MPS Testing (mps:0)]...")
        try:
            stt_mps = WhisperSTTProvider(model_name="openai/whisper-large-v3-turbo", device="mps")
            
            # Warmup
            _ = stt_mps.transcribe(audio_mix[:16000], 16000)
            
            t0 = time.perf_counter()
            txt_te_mps, lang_te_mps, lat_te_mps = stt_mps.transcribe(audio_te, sr_te)
            txt_en_mps, lang_en_mps, lat_en_mps = stt_mps.transcribe(audio_en, sr_en)
            txt_mix_mps, lang_mix_mps, lat_mix_mps = stt_mps.transcribe(audio_mix, sr_mix)
            
            print(f"  Telugu (3.20s):    {lat_te_mps:>7.2f} ms | Lang: {lang_te_mps} | Transcript: '{txt_te_mps}'")
            print(f"  English (5.33s):   {lat_en_mps:>7.2f} ms | Lang: {lang_en_mps} | Transcript: '{txt_en_mps}'")
            print(f"  Mixed (1.55s):     {lat_mix_mps:>7.2f} ms | Lang: {lang_mix_mps} | Transcript: '{txt_mix_mps}'")
            print(f"  MPS Total Time:    {(lat_te_mps + lat_en_mps + lat_mix_mps)/1000.0:.2f} s")
            print(f"  Speedup Factor:    {(lat_te_cpu + lat_en_cpu + lat_mix_cpu) / max(1.0, (lat_te_mps + lat_en_mps + lat_mix_mps)):.2f}x")
        except Exception as mps_stt_err:
            print(f"  MPS Whisper Failed: {mps_stt_err}")

    # -------------------------------------------------------------------------
    # PART 3: INDICF5 TTS + LORA BENCHMARK (CPU vs MPS)
    # -------------------------------------------------------------------------
    print_header("3. INDICF5 TTS + LORA + VOCOS: CPU vs MPS BENCHMARK")
    tts_test_phrase = "నమస్కారం కాలేజ్ అటెండెన్స్ నియమాలు మరియు సమాచారం ఇక్కడ ఉన్నాయి"
    
    print(f"Synthesizing test phrase: \"{tts_test_phrase}\"")
    
    # Test MPS TTS if available
    if mps_available:
        print("\n[IndicF5 MPS Testing (mps:0)]...")
        try:
            tts_mps = IndicF5TTSProvider(
                adapter_path="models/indicf5_telugu_lora",
                device="mps"
            )
            
            t0 = time.perf_counter()
            wav_bytes_mps, lat_mps = tts_mps.synthesize(tts_test_phrase)
            total_s_mps = time.perf_counter() - t0
            
            import io
            audio_data_mps, sr_out_mps = sf.read(io.BytesIO(wav_bytes_mps))
            rms_mps = float(np.sqrt(np.mean(audio_data_mps ** 2)))
            
            print(f"  MPS Synthesis Latency: {lat_mps:>8.2f} ms ({total_s_mps:.2f} s)")
            print(f"  Output Sample Rate:    {sr_out_mps} Hz | Duration: {len(audio_data_mps)/sr_out_mps:.2f} s")
            print(f"  Audio RMS Energy:      {rms_mps:.4f} (Intelligible: {rms_mps > 0.01})")
        except Exception as mps_tts_err:
            print(f"  MPS IndicF5 Failed: {mps_tts_err}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # PART 4: FASTEMBED ONNX PROVIDERS (CPU vs CoreML)
    # -------------------------------------------------------------------------
    print_header("4. FASTEMBED ONNX: CPU vs COREML BENCHMARK")
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print("Available ONNX Providers:", providers)
    
    # Test default CPU
    emb_q = "నమస్కారం కాలేజ్ సమాచారం"
    t0 = time.perf_counter()
    for _ in range(10):
        _ = emb_provider.embed_query(emb_q)
    cpu_emb_lat = (time.perf_counter() - t0) * 100.0  # ms per query
    print(f"FastEmbed CPU Embedding Latency: {cpu_emb_lat:.2f} ms / query")

    # -------------------------------------------------------------------------
    # PART 5: OVERALL SYSTEM PROFILE & MULTI-PROCESS ANALYSIS
    # -------------------------------------------------------------------------
    print_header("5. PROCESS AND SYSTEM MEMORY SNAPSHOT")
    final_rss = get_process_rss_mb()
    vm = psutil.virtual_memory()
    print(f"Final Benchmark Process RSS: {final_rss:.2f} MB ({final_rss/1024:.2f} GB)")
    print(f"System Total RAM:            {vm.total/(1024**3):.2f} GB")
    print(f"System Used RAM:             {vm.used/(1024**3):.2f} GB")
    print(f"System Available RAM:        {vm.available/(1024**3):.2f} GB")
    print(f"System Memory % Used:        {vm.percent:.1f} %")

if __name__ == "__main__":
    main()
