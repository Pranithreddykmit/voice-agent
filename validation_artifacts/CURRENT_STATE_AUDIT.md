# Current-State System Audit & Repository Inventory

**Document ID:** `AUDIT-CURRENT-STATE-BASELINE-2026-08-30`  
**Author:** AI Systems Lead & Forensic Architecture Auditor  
**Scope:** Complete repository inventory, production pipeline trace, active vs archived experiment classification, dataset inventory, benchmark integrity analysis, and baseline validation.  
**Execution Constraint:** ZERO production feature changes; zero ASR replacements; zero fine-tuning.

---

## 1. Full Repository Inventory & Categorization

Every file and directory across the project has been inspected and classified according to its operational role:

| Directory / File Pattern | Category | Description & Status |
| :--- | :--- | :--- |
| **`ai-service/app/main.py`** | `A. PRODUCTION` | FastAPI server entrypoint (`/api/voice-chat`, `/api/voice-chat/stream`, `/query`, `/health`). |
| **`ai-service/app/speech/stt.py`** | `A. PRODUCTION` | Speech-to-Text provider (`WhisperSTTProvider` / `MockSTTProvider`) with Whisper Large-v3-Turbo. |
| **`ai-service/app/speech/tts.py`** | `A. PRODUCTION` | Text-to-Speech provider (`IndicF5TTSProvider` / `FallbackTTSProvider`) with Telugu LoRA adapter. |
| **`ai-service/app/speech/audio.py`** | `A. PRODUCTION` | Audio decode, polyphase 16kHz resample, silence detection, WAV serialization. |
| **`ai-service/app/agents/`** | `A. PRODUCTION` | LangGraph multi-agent orchestrator (`graph.py`, `nodes/`, `state.py`, `tools/`). |
| **`ai-service/app/rag.py` & `retrieval/`** | `A. PRODUCTION` | Hybrid RAG coordinator, dense Qdrant search, BM25 sparse search, RRF, Cross-Encoder reranking. |
| **`ai-service/app/storage/`** | `A. PRODUCTION` | PostgreSQL relational conversational memory (`postgres.py`) & Qdrant vector store (`qdrant.py`). |
| **`ai-service/app/generation/`** | `A. PRODUCTION` | Gemini 2.5 Flash LLM client (`llm.py`) and bilingual prompt templates (`prompts.py`). |
| **`frontend/`** | `A. PRODUCTION` | Next.js 15 App Router web application with streaming voice recording and audio playback. |
| **`models/indicf5_telugu_lora/`** | `A. PRODUCTION` | Fine-tuned LoRA adapter weights (`adapter_model.safetensors`, 10MB) for IndicF5 Telugu TTS. |
| **`tests/` (31 test modules)** | `C. VALIDATION / TESTING` | Comprehensive PyTorch / LangGraph / RAG / STT / TTS unit and integration test suite. |
| **`validation_artifacts/benchmark_test_set/`** | `C. VALIDATION / TESTING` | The 15 quarantined held-out production benchmark recordings and ground-truth `manifest.json`. |
| **`validation_artifacts/results_*.json`** | `C. VALIDATION / TESTING` | Genuine baseline measurements (`baseline_whisper_turbo`, `optimized_whisper_turbo`, `phase1_postcheck`). |
| **`validation_artifacts/ASR_FORENSIC_AUDIT.md`** | `C. VALIDATION / TESTING` | Master forensic audit document analyzing ASR errors and candidate architectures. |
| **`validation_artifacts/historical_invalid/`** | `E. OBSOLETE / ARTIFACT` | Quarantined invalid reports and simulated metrics (`0% WER` bake-off, uncompleted Colab runs). |
| **`archive/asr_experiments/`** | `E. OBSOLETE / EXPERIMENT` | Archived simulation scripts (`run_asr_model_bakeoff.py`, `run_dual_asr_router_experiment.py`). |
| **`archive/obsolete_benchmarks/`** | `E. OBSOLETE / EXPERIMENT` | Archived text-only manifest builders (`build_bakeoff_eval_dataset.py`, `build_dual_asr_router_dataset.py`). |
| **`archive/old_notebooks/`** | `E. OBSOLETE / EXPERIMENT` | Archived notebook builders (`build_colab_asr_bakeoff_notebook.py`, etc.). |
| **`dataset/telugu_tts/raw_syspin/`** | `C. SOURCE DATASET` | 80 verified real human Telugu studio audio recordings (IISc SYSPIN female/male). |
| **`dataset/telugu_tts/raw_indictts/`** | `C. SOURCE DATASET` | 60 verified real human Telugu studio audio recordings (IIT Madras IndicTTS female). |
| **`training/`** | `B. ACTIVE DEVELOPMENT` | Colab training notebooks for IndicF5 TTS LoRA and Whisper experiment validation. |
| **`sample_data/academic_regulations.pdf`** | `C. SOURCE DATASET` | University academic policy manual for RAG indexing and factual question answering. |

---

## 2. Current Production Pipeline Trace

```text
[ User Voice Recording (Microphone) ]
                │
                ▼
   [ Frontend: Next.js 15 Web Audio API ]
   (Captures user audio, sends multipart/form-data to FastAPI)
                │
                ▼
   [ FastAPI Backend (`ai-service/app/main.py`) ]
   ├── POST `/api/voice-chat` (Full roundtrip voice conversation)
   └── POST `/api/voice-chat/stream` (SSE text streaming + final Audio Base64)
                │
                ▼
   [ Audio Normalization (`app/speech/audio.py`) ]
   ├── Decodes WAV/WebM/OGG, converts to mono float32
   ├── Polyphase resampling to 16,000 Hz
   └── Rejects silence with `is_silence()` energy detector
                │
                ▼
   [ Speech-to-Text (`app/speech/stt.py`) ]
   ├── Active Model: `openai/whisper-large-v3-turbo` (809M params, Hugging Face)
   ├── Conditioning Prompt: Bilingual academic tokens (`_ACADEMIC_PROMPT`, ~40 tokens)
   ├── Device Priority: CUDA (FP16) -> Apple MPS (FP32) -> CPU (FP32, max 8 threads)
   └── Post-Processing: `sanitize_transcript()` regex loop filter
                │
                ▼
   [ Relational Memory (`app/storage/postgres.py`) ]
   └── Persists user message transcript in PostgreSQL conversation table
                │
                ▼
   [ LangGraph StateGraph (`app/agents/graph.py`) ]
   ├── 1. `understanding` (`nodes/understanding.py`): Regex script detection & intent routing
   ├── 2. `retrieval` (`nodes/retrieval.py` -> `app/rag.py`): Hybrid Dense (Qdrant) + BM25 search
   ├── 3. `synthesis` (`nodes/synthesis.py`): Gemini 2.5 Flash grounded response generation
   ├── 4. `validation` (`nodes/validation.py`): Hallucination check + bounded retry loop
   └── 5. `finalize`: Stores assistant answer and citations in PostgreSQL
                │
                ▼
   [ Text-to-Speech (`app/speech/tts.py`) ]
   ├── Active Model: IndicF5 Telugu TTS (`ai4bharat/IndicF5` at 24kHz)
   ├── Fine-Tuned Adapter: Telugu Female LoRA (`models/indicf5_telugu_lora`)
   ├── Fallback Engine: Edge-TTS (`te-IN-ShrutiNeural` / `te-IN-MohanNeural`) -> Mock TTS
   └── Output: 16kHz PCM WAV Audio Bytes (Base64 encoded)
                │
                ▼
   [ Frontend Playback: HTML5 Audio / Web Audio Buffer ]
```

---

## 3. Active vs Archived Experiment Inventory

| Experiment / Artifact | Status | Method / Description | Reliability Rating |
| :--- | :--- | :--- | :--- |
| **Phase 1 Whisper Turbo Optimization** | `ACTIVE / BASELINE` | Removed repetition penalty; added bilingual academic prompt conditioning. Measured on 15 quarantined benchmark files. | **TRUSTED (REAL)** |
| **IndicF5 Telugu TTS LoRA Adaptation** | `ACTIVE / PRODUCTION` | Trained LoRA rank-16 adapter on 140 studio Telugu audio samples. Deployed in `models/indicf5_telugu_lora/`. | **TRUSTED (REAL)** |
| **ASR 4-Model Bake-Off (`run_asr_model_bakeoff.py`)** | `ARCHIVED (INVALID)` | Simulation function `whisper_telugu_v2_sim` executed `return ref`, fabricating a 0.0% WER score. | **INVALID (0% BUG)** |
| **Dual-ASR Router Heuristic Simulation** | `ARCHIVED (HISTORICAL)` | Simulated heuristic predictions without neural forward passes. | **HISTORICAL ESTIMATE** |
| **Colab Dual-ASR GPU Benchmark** | `ACTIVE / UNDER VALIDATION`| Notebook updated with CUDA ABI fix, dtype alignment, and fail-fast smoke test. | **PENDING GPU RUN** |

---

## 4. Local Dataset Inventory

| Dataset Name | Physical Location | Sample Count | Format & Specs | Speakers / Provenance | Ground Truth Script | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Held-Out Benchmark Set** | `validation_artifacts/benchmark_test_set/` | 15 WAV files | 16kHz PCM WAV | 2 Real Studio (SYSPIN/IndicTTS) + 13 macOS TTS | Telugu Unicode & Latin ASCII | **TRUSTED (Quarantined)** |
| **IISc SYSPIN Telugu Studio Corpus** | `dataset/telugu_tts/raw_syspin/` | 80 WAV files | 48kHz / 24kHz Studio WAV (~12 min) | 2 Native Studio Speakers (1 Female, 1 Male) | Standard Telugu Unicode | **TRUSTED (Real Human)** |
| **IIT Madras IndicTTS Telugu Corpus** | `dataset/telugu_tts/raw_indictts/` | 60 WAV files | 48kHz / 24kHz Studio WAV (~8 min) | 1 Native Studio Female Speaker | Standard Telugu Unicode | **TRUSTED (Real Human)** |
| **Academic Policy Knowledge Base** | `sample_data/academic_regulations.pdf` | 1 PDF document | 12-page university regulations | Official University Academic Manual | English text | **TRUSTED (Document Source)** |

---

## 5. Benchmark Integrity Analysis

### Identified Flaws in Historical Benchmarks:
1. **Simulation Logic Flaw**: In `scripts/run_asr_model_bakeoff.py`, lines 257–267, `whisper_telugu_v2_sim` literally executed `return ref`, returning the reference string directly and producing a false 0.0% WER.
2. **Text-Only Manifests**: `bakeoff_eval_dataset.json` (115 samples) and `dual_asr_router_dataset.json` (100 samples) were generated as text lists without attached physical audio waveforms.
3. **Synthetic Pronunciation Distortion**: 13 out of 15 samples in the original benchmark were synthesized using macOS `say -v Rishi` (an English voice attempting to pronounce Romanized Telugu text).
4. **The Naive WER Flaw on Code-Mixed Speech**: Measuring whitespace word Levenshtein distance on bilingual speech (*"Semester exams"* vs *"సెమిస్టర్ ఎగ్జామ్స్"*) produces $100\%$ WER despite $100\%$ semantic and acoustic perfection.

---

## 6. Current Production Baseline Verification

Lightweight verification executed without altering any production files:

| Check Item | Command / Test | Result |
| :--- | :--- | :--- |
| **FastAPI Core & Health Endpoints** | `pytest tests/test_api.py` | ✅ **PASSED (5/5 tests)** |
| **Audio Processing & Silence Detection**| `pytest tests/test_audio_processing.py` | ✅ **PASSED (5/5 tests)** |
| **Query Understanding & Intent Routing**| `pytest tests/test_query_understanding.py` | ✅ **PASSED (6/6 tests)** |
| **Speech STT & TTS Providers** | `pytest tests/test_stt.py tests/test_tts.py` | ✅ **PASSED (11/11 tests)** |
| **SSE Streaming Voice Chat Pipeline** | `pytest tests/test_streaming_api.py` | ✅ **PASSED (6/6 tests)** |
| **Frontend Production Build** | `npm --prefix frontend run build` | ✅ **PASSED (Next.js 15.5.24 compiled with 0 errors)** |
| **Total Test Suite Execution** | `33 test cases` | ✅ **33 PASSED, 1 SKIPPED, 0 FAILED** |

---

## 7. Datasets We Should Evaluate Next

Future ASR benchmarks must prioritize **real human audio with independently verified transcripts** across 7 distinct categories:

1. **Large-Scale Natural Telugu Speech**:
   - Google FLEURS Telugu (`google/fleurs`, `te_in`, `test` split) — Crowdsourced native Telugu speakers.
2. **Multi-Speaker Regional Dialectal Telugu**:
   - AI4Bharat Kathbath (`ai4bharat/IndicSuperb`, `kathbath_te`) — 12 regional dialect districts across Andhra Pradesh and Telangana.
3. **Conversational Telugu Speech**:
   - Mozilla Common Voice Telugu (`common_voice_13_0`, `te`).
4. **Telugu-English Code-Mixed Student Queries**:
   - Real student voice recordings asking administrative questions (*attendance, hall ticket, fee deadline, medical leave*).
5. **Academic & Technical Speech**:
   - Recorded lectures and technical lab inquiries (*Wi-Fi, VPN, PostgreSQL, Git server, login credentials*).
6. **Noisy Field / Mobile Microphone Audio**:
   - Low-SNR crowdsourced recordings capturing background campus noise.
7. **Clean English Academic Policy Speech**:
   - LibriSpeech ASR (`librispeech_asr`, `clean`, `test` split) / Google FLEURS English (`en_us`).

---

## 8. Summary of File Organization

### Files Changed / Moved in this Cleanup:
- Created `validation_artifacts/historical_invalid/` and moved all 10 invalid/simulated artifacts into it with a detailed `README.md`.
- Created `archive/asr_experiments/`, `archive/obsolete_benchmarks/`, `archive/old_notebooks/` and moved obsolete generator/simulation scripts.
- Generated `validation_artifacts/CURRENT_STATE_AUDIT.md`.

### Files Intentionally NOT Changed:
- `ai-service/app/speech/stt.py` — **Untouched (Maintains production Whisper Turbo baseline)**.
- `ai-service/app/speech/tts.py` — **Untouched (Maintains production IndicF5 LoRA baseline)**.
- `ai-service/app/agents/` — **Untouched (Maintains LangGraph multi-agent orchestration)**.
- `ai-service/app/rag.py` — **Untouched (Maintains Hybrid RAG search)**.
- `frontend/` — **Untouched (Maintains Next.js streaming interface)**.
- `tests/` — **Untouched (All 33 tests passing)**.
