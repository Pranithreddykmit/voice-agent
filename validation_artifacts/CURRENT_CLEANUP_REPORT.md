# Repository Cleanup & Stabilization Report

**Document ID:** `CURRENT-CLEANUP-REPORT-2026-08-31`  
**Status:** COMPLETED & VERIFIED  
**Operation:** Archive and clean all abandoned ASR experimental artifacts while freezing the verified production baseline.

---

## 1. Executive Summary

The repository has been thoroughly cleaned and stabilized. All obsolete, speculative, and intermediate ASR experimentation files (old bake-offs, simulated outputs, router experiments, obsolete Colab notebooks, and temporary evaluation logs) have been safely relocated into the timestamped archive directory:
`archive/cleanup_2026-08-31/`

**Zero production source code was modified.**  
**Zero historical evidence was lost (all preserved under `archive/`).**

---

## 2. Active Verified Production Baseline

The production application architecture is frozen and verified:

```text
[Audio Input] 
      │
      ▼
[Whisper Large-v3-Turbo STT] (Prompt-conditioned, openai/whisper-large-v3-turbo)
      │
      ▼
[Query Expansion Layer] (Non-destructive bilingual keyword expansion)
      │
      ▼
[LangGraph Multi-Agent Orchestrator] (Intent classification & validation)
      │
      ▼
[Hybrid RAG Engine] (Dense E5 + Sparse BM25 + Reciprocal Rank Fusion + BGE Cross-Encoder Reranker)
      │
      ▼
[Gemini LLM / Grounded Response Generator]
      │
      ▼
[IndicF5 Neural TTS + Telugu LoRA Adapter] (models/indicf5_telugu_lora, 24kHz PCM WAV)
      │
      ▼
[Next.js TypeScript Frontend] (SSE Streaming, Audio Player, Document Management)
```

* **ASR Status:** Experiments **FROZEN / NOT IN PRODUCTION**.
* **TTS Status:** **IndicF5 + Telugu LoRA** (`models/indicf5_telugu_lora`) active and verified.
* **Backend Test Suite:** **97 Unit Tests Passing** (13 Docker/live tests cleanly skipped).
* **Frontend Build:** **Next.js Production Build Passing** (0 type errors, 0 lint errors).

---

## 3. Detailed Inventory of Cleaned Items

### 3.1 Files Archived to `archive/cleanup_2026-08-31/` (70 Items Total):

#### A. `validation_artifacts/` (31 Items Archived):
- `historical_invalid/` (Entire folder containing historical simulated/invalid bake-off outputs)
- `asr_bakeoff_validation_report.md`
- `ASR_CURRENT_STATE.md`
- `ASR_DATASET_FINAL_PLAN.md`
- `ASR_DATASET_MANIFEST_PLAN.json`
- `ASR_DATASET_RESEARCH.md`
- `ASR_DATASET_SAMPLE_VERIFICATION.md`
- `ASR_ERROR_IMPACT_ANALYSIS.json` & `.md`
- `ASR_FINAL_DECISION.md`
- `ASR_FORENSIC_AUDIT.md`
- `EXPERIMENT_CLEANUP.md`
- `WHISPER_TURBO_95_REAL_BASELINE.md`
- `WHISPER_TURBO_BASELINE.md`
- `results_baseline_whisper_turbo.json`
- `results_lora_heldout_testset.json`
- `results_optimized_whisper_turbo.json`
- `results_phase1_postcheck.json`
- `whisper_turbo_95_real_baseline.json`
- `whisper_turbo_real_baseline.json`
- Intermediate WAV snippets: `e2e_query_attendance.wav`, `stt_en_long.wav`, `stt_en_norm.wav`, `stt_mixed_1.wav`, `stt_mixed_2.wav`, `stt_te_norm.wav`, `stt_te_short.wav`, `tts_english_reply.wav`, `tts_greeting.wav`, `tts_multi_sentence.wav`, `tts_negative_rag.wav`

#### B. `training/` (11 Items Archived):
- `colab_acquire_and_eval_95_real.ipynb` & `.py`
- `colab_asr_bakeoff.ipynb`
- `colab_asr_final_bakeoff.ipynb` & `.py`
- `colab_dataset_pipeline.py`
- `colab_dual_asr_router_experiment.ipynb`
- `colab_stt_forced_telugu_ab.ipynb` & `.py`
- `train_whisper_lora.py`
- `whisper_lora_colab.ipynb`

#### C. `scripts/` (15 Items Archived):
- `build_asr_benchmark_testset.py`
- `diagnose_stt_failures.py`
- `evaluate_whisper_lora_testset.py`
- `evaluate_whisper_turbo_real_baseline.py`
- `prepare_whisper_lora_dataset.py`
- `run_asr_error_impact_analysis.py`
- `run_asr_mms_benchmark.py`
- `run_asr_model_benchmark.py`
- `run_asr_optimized_whisper.py`
- `run_asr_phase1_postcheck.py`
- `run_asr_whisper_small.py`
- `run_forced_telugu_ab_experiment.py`
- `run_real_asr_colab_benchmark.py`
- `run_realistic_performance_validation.py`
- `validate_vasista_on_real_audio.py`

#### D. `evaluation/` (12 Items Archived):
- `agent_benchmark_results.json`
- `benchmark_results.json`
- `evaluate_lora_comparative.py`
- `evaluate_phase7.py`
- `phase6d_real_results.json` & `_transcriptions.json`
- `phase6d_results.json`
- `phase7_full_results.json` & `_test_results.json`
- `phase7_transcriptions.json` & `_tts_benchmark.json`
- `speech_benchmark_results.json`

#### E. `dataset/` (1 Item Archived):
- `dataset/whisper_lora/` (Entire folder)

---

### 3.2 Disposable Cache Files Deleted (25 Items Total):
- `15` `__pycache__` directories across the repository.
- `10` `.DS_Store` macOS metadata files.

---

## 4. Current Clean Repository Tree

```text
.
├── .env.example
├── README.md
├── START_GUIDE.md
├── docker-compose.yml
├── ai-service/                   # FastAPI Backend & Agent Core
│   ├── app/
│   │   ├── agents/               # LangGraph Orchestrator & State Workflow
│   │   ├── embeddings/           # Multilingual E5 Embedding Providers
│   │   ├── generation/           # Gemini LLM & Prompts
│   │   ├── ingestion/            # PDF Parser & Telugu Text Chunker
│   │   ├── retrieval/            # BM25Plus, RRF, Query Expansion, BGE Reranker
│   │   ├── speech/               # Whisper Turbo STT & IndicF5 TTS
│   │   ├── storage/              # PostgreSQL Memory & Qdrant Client
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── rag.py
│   │   └── schemas.py
│   └── requirements.txt
├── frontend/                     # Next.js 15 TypeScript Frontend UI
├── models/
│   └── indicf5_telugu_lora/      # Active Telugu TTS LoRA Weights (Epoch 44)
├── dataset/
│   └── telugu_tts/               # QUARANTINED (Preserved for TTS, NOT for ASR)
│       └── README_QUARANTINE.md
├── validation_artifacts/
│   ├── benchmark_test_set/       # 15-sample Quarantined Reference Benchmark
│   ├── asr_real_eval_manifest.json # 95-sample Real Human Benchmark Manifest
│   ├── CURRENT_STATE_AUDIT.md
│   ├── STT_FAILURE_FORENSIC_ANALYSIS.json & .md
│   ├── STT_FORCED_TELUGU_AB_COMPARISON.json & .md
│   └── CURRENT_CLEANUP_REPORT.md # This report
├── training/
│   ├── indicf5_lora_colab.ipynb  # Active IndicF5 Telugu TTS LoRA Notebook
│   ├── inspect_lora.py           # LoRA Weight Inspector
│   └── train_lora.py             # IndicF5 LoRA Training Script
├── evaluation/
│   ├── README.md
│   ├── benchmark_tts.py
│   ├── dataset.json
│   ├── evaluate.py
│   ├── evaluate_agent.py
│   ├── evaluate_speech.py
│   ├── metrics.py
│   └── speech_dataset.json
├── scripts/
│   ├── benchmark_hardware_acceleration.py
│   ├── benchmark_streaming.py
│   ├── diagnose_single_tts_live.py
│   ├── diagnose_tts_memory.py
│   ├── diagnose_tts_memory_mps.py
│   ├── generate_sample_pdf.py
│   ├── test_direct_tts.py
│   ├── test_granular_tts.py
│   ├── verify_live_app_memory.py
│   └── verify_speech_subsystem.py
├── tests/                        # 110 Comprehensive Backend Unit & Integration Tests
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PROJECT_STATUS.md
│   └── RESEARCH.md
└── archive/                      # Preserved Historical Experiments
    └── cleanup_2026-08-31/
```

---

## 5. Verification & Test Status

1. **Backend Tests:** **97 Passed**, 13 Skipped (Docker/CUDA guards), **0 Failures** in 34.96s.
2. **Frontend Build:** `npm run build` compiled successfully in **701ms** (0 type errors, 0 lint errors).
3. **Production Files Modified:** **0 Files**.

```
================================================================================
REPOSITORY CLEANUP COMPLETE — REPOSITORY IN CLEAN & STABLE STATE
================================================================================
```
