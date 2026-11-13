# ASR Current State & Forensic Verification Audit

**Document ID:** `ASR-CURRENT-STATE-2026-08-30`  
**Status:** COMPLETED EMPIRICAL REPOSITORY AUDIT (Phase 0)  
**Execution Constraint:** ZERO production code modified; ZERO fine-tuning; ZERO synthetic speech benchmark generation.

---

## 1. Honest State Table

| Component | Current Implementation | Verified? | Evidence | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Speech-to-Text (STT)** | `openai/whisper-large-v3-turbo` (809M params) via Hugging Face `AutoProcessor` & `AutoModelForSpeechSeq2Seq` in `WhisperSTTProvider` | ✅ **VERIFIED** | [`ai-service/app/speech/stt.py:90-290`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/speech/stt.py#L90-L290), `test_stt.py` (all tests passing) | **PRODUCTION ACTIVE** |
| **STT Device & Precision** | Auto-detection: CUDA (`float16`), Apple MPS (`float32`), CPU (`float32`, thread pool capped at $\min(8, N_{\text{cores}})$) | ✅ **VERIFIED** | [`ai-service/app/speech/stt.py:119-136`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/speech/stt.py#L119-L136) | **PRODUCTION ACTIVE** |
| **STT Prompt & Decoding** | `task="transcribe"`, `prompt_ids=_ACADEMIC_PROMPT` (~40 tokens bilingual vocabulary), unforced multilingual language auto-detection, `max_new_tokens=256`, `sanitize_transcript()` regex loop filter | ✅ **VERIFIED** | [`ai-service/app/speech/stt.py:18-24, 144-148, 241-258`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/speech/stt.py#L18-L24) | **PRODUCTION ACTIVE** |
| **STT $\rightarrow$ RAG Pipeline** | Transcript from `stt.transcribe()` is persisted to PostgreSQL conversation history, then routed into LangGraph `understanding` $\rightarrow$ `retrieval` (Hybrid Dense Qdrant + Sparse BM25 + RRF + BGE Reranker) $\rightarrow$ `synthesis` (Gemini Flash) | ✅ **VERIFIED** | [`ai-service/app/main.py:527-550`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/main.py#L527-L550), [`ai-service/app/agents/graph.py`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/agents/graph.py) | **PRODUCTION ACTIVE** |
| **Text-to-Speech (TTS)** | `ai4bharat/IndicF5` Flow-Matching Diffusion DiT (24,000 Hz) with Vocos neural vocoder in `IndicF5TTSProvider` | ✅ **VERIFIED** | [`ai-service/app/speech/tts.py:147-472`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/speech/tts.py#L147-L472), `test_tts.py` | **PRODUCTION ACTIVE** |
| **TTS LoRA Adapter** | `models/indicf5_telugu_lora` (Epoch 44 Best Checkpoint, 2.52M weights, 10.1 MB safetensors) | ✅ **VERIFIED** | [`models/indicf5_telugu_lora/adapter_model.safetensors`](file:///Users/sathvikreddy/Documents/projects/tel/models/indicf5_telugu_lora/adapter_model.safetensors), [`ai-service/app/speech/tts.py:261-304`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/speech/tts.py#L261-L304) | **PRODUCTION ACTIVE** |
| **Local Benchmark Test Set** | 15 WAV files in `validation_artifacts/benchmark_test_set/` (2 Real Studio + 13 macOS synthetic TTS) | ✅ **VERIFIED** | [`validation_artifacts/benchmark_test_set/manifest.json`](file:///Users/sathvikreddy/Documents/projects/tel/validation_artifacts/benchmark_test_set/manifest.json) | **QUARANTINED BENCHMARK** |
| **Local Real Studio Audio** | 140 Real Studio WAV files (80 IISc SYSPIN + 60 IIT Madras IndicTTS) in `dataset/telugu_tts/raw_*` | ✅ **VERIFIED** | [`dataset/telugu_tts/raw_syspin/`](file:///Users/sathvikreddy/Documents/projects/tel/dataset/telugu_tts/raw_syspin/), [`dataset/telugu_tts/raw_indictts/`](file:///Users/sathvikreddy/Documents/projects/tel/dataset/telugu_tts/raw_indictts/) | **REAL HUMAN DATA (QUARANTINED FROM ASR TRAINING)** |
| **Bake-Off Simulation Report** | Stated Vasista22 at 0.0% WER across 115 samples via `return ref` | ❌ **EXPOSED (INVALID)** | `run_asr_model_bakeoff.py:257-267`, [`validation_artifacts/historical_invalid/`](file:///Users/sathvikreddy/Documents/projects/tel/validation_artifacts/historical_invalid/) | **QUARANTINED / INVALID** |
| **Dual-ASR Router Heuristic** | 100-sample text-only manifest with simulated acoustic heuristic predictions | ⚠️ **HISTORICAL ESTIMATE** | [`validation_artifacts/historical_invalid/asr_router_benchmark_report.md`](file:///Users/sathvikreddy/Documents/projects/tel/validation_artifacts/historical_invalid/asr_router_benchmark_report.md) | **QUARANTINED / HISTORICAL** |
| **AI4Bharat IndicVoices** | Gated on Hugging Face (`ai4bharat/IndicVoices`, `gated: auto`, ~73 hrs Telugu). NOT on local disk. | ⚠️ **DOCUMENTED ONLY** | Verified against Hugging Face API & paper arXiv:2403.01926 | **ACCESSIBLE VIA HF TOKEN STREAMING** |
| **AI4Bharat Kathbath** | Gated on Hugging Face (`ai4bharat/IndicSuperb`, `gated: auto`, ~100 hrs Telugu). NOT on local disk. | ⚠️ **DOCUMENTED ONLY** | Verified against Hugging Face API & paper arXiv:2207.00696 | **ACCESSIBLE VIA HF TOKEN STREAMING** |
| **Google FLEURS** | Open CC-BY-4.0 on Hugging Face (`google/fleurs`, `te_in` 12.5 hrs, `en_us` 12.0 hrs). NOT on local disk. | ⚠️ **DOCUMENTED ONLY** | Verified against Hugging Face Hub & paper arXiv:2205.12446 | **ACCESSIBLE (STREAMING / SLICES)** |
| **CoSTA Benchmark** | Open Research on GitHub (`csalt-research/CoSTA`, COLING 2025, ~15 hrs). NOT on local disk. | ⚠️ **DOCUMENTED ONLY** | Verified against GitHub repo & COLING 2025 paper | **ACCESSIBLE FOR TEST ONLY** |
| **Sarvam AI Saaras:v3** | Commercial Cloud REST/WebSocket API (`mode="codemix"`). No API key present in `.env`. | ❌ **UNAVAILABLE** | Checked `.env` and codebase; `SARVAM_API_KEY` is not present | **UNAVAILABLE (NO CREDENTIALS)** |

---

## 2. Detailed Technical Audit Findings

### 2.1 STT Loading & Audio Ingestion Flow
1. **Audio Input**: Received as multipart file via `/api/stt` or `/api/voice-chat`.
2. **Audio Preprocessing** ([`app/speech/audio.py`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/speech/audio.py)):
   - Decodes audio format (WAV/WebM/OGG) to 1D mono float32 array.
   - Resamples to 16,000 Hz using `scipy.signal.resample_poly`.
   - Rejects silence using root-mean-square (RMS) energy threshold.
3. **Inference Execution** ([`app/speech/stt.py`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/speech/stt.py)):
   - Model: `openai/whisper-large-v3-turbo` (809M params) loaded via Hugging Face `AutoProcessor` and `AutoModelForSpeechSeq2Seq`.
   - Dtype: `float16` on CUDA, `float32` on Apple MPS and CPU.
   - Vocabulary Conditioning: `prompt_ids` encodes `_ACADEMIC_PROMPT` ("attendance, semester exams, hostel, library, fee, medical leave, హాజరు, పరీక్షలు, ఫీజు, దరఖాస్తు").
   - Decoding: `task="transcribe"`, `language=None` (unforced auto-detection).
   - Post-Processing: `sanitize_transcript()` collapses character and word repetition spirals.
4. **Handoff to RAG & Storage** ([`app/main.py`](file:///Users/sathvikreddy/Documents/projects/tel/ai-service/app/main.py)):
   - Transcript is persisted to PostgreSQL message history (`role="user"`).
   - Passed as `query` to `AgentOrchestrator.run()`.
   - LangGraph `understanding` node classifies intent and language (`telugu`, `english`, `mixed`).
   - If `knowledge_query`, `retrieval` node invokes `RAGPipeline.hybrid_search(query)` against Qdrant (`intfloat/multilingual-e5-large`) and sparse BM25 index with Reciprocal Rank Fusion ($k=60$) and Cross-Encoder reranking (`BAAI/bge-reranker-base`).
   - `synthesis` node invokes Google Gemini Flash to generate grounded Telugu/English response.
   - `validation` node verifies citations and faithfulness.
   - `finalize` node hands response text to `IndicF5TTSProvider.synthesize()`.

### 2.2 Local Physical Datasets vs Documented Datasets
* **Physical Local Audio Files**:
  1. `validation_artifacts/benchmark_test_set/`: 15 WAV files (Quarantined benchmark).
  2. `dataset/telugu_tts/raw_syspin/`: 80 WAV files (IISc SYSPIN studio Telugu).
  3. `dataset/telugu_tts/raw_indictts/`: 60 WAV files (IIT Madras IndicTTS studio Telugu).
* **Train-Test Leakage Warning**:
  - `A1_te_human_syspin.wav` in the 15-sample benchmark is identical to `raw_syspin/syspin_f_0002.wav`.
  - `A2_te_human_indictts.wav` in the 15-sample benchmark is identical to `raw_indictts/indictts_f_0011.wav`.
  - Therefore, `dataset/telugu_tts/` is strictly quarantined from ASR training.
* **Documented-Only Public Datasets (Not stored on local disk)**:
  - `ai4bharat/IndicVoices` (~73 hrs Telugu, gated on Hugging Face).
  - `ai4bharat/Kathbath` (~100 hrs Telugu, gated on Hugging Face).
  - `google/fleurs` (12.5 hrs Telugu, open on Hugging Face).
  - `csalt-research/CoSTA` (~15 hrs Telugu-English code-switched ST, on GitHub).
  - `Sarvam Saaras:v3` (Commercial API, no API key available).

---

## 3. Executive Assessment Summary

```
================================================================================
CURRENT STATE AUDIT VERDICT:
- Production STT: Whisper Large-v3-Turbo (Functional, Prompt-Conditioned)
- Production TTS: IndicF5 + Telugu LoRA (Verified, Fully Functional)
- Production RAG: LangGraph + Hybrid Qdrant/BM25 + Gemini (Verified, 33/33 Tests Passing)
- Verified Local Real Audio: 140 Studio WAV files (Quarantined from ASR training)
- Quarantined Invalid Artifacts: 0% WER bake-off bug and text-only simulated manifests
- Cloud Datasets Status: IndicVoices / Kathbath / FLEURS / CoSTA accessible in Colab; Sarvam unavailable (no key).
================================================================================
```
