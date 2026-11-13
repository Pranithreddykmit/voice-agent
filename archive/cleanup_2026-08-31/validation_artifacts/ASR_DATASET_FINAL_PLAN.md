# ASR Dataset Reality Check & Final Evaluation Plan

**Document ID:** `ASR-DATASET-FINAL-PLAN-2026-08-30`  
**Status:** COMPLETED REALITY AUDIT (Phase 2)  
**Core Directive:** Real human audio ONLY. No synthetic TTS ground truth. No massive multi-GB local downloads.

---

## 1. Master Dataset Reality Table

| Dataset | Real Audio? | Telugu | Code-Mix | English | Domain Relevance | License / Access | Training / Eval Role | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AI4Bharat IndicVoices** (`ai4bharat/IndicVoices`) | ✅ **YES** (150+ native speakers, 16kHz WAV) | ✅ **YES** (~73 hrs) | ✅ **YES** (Spontaneous splits have natural loanwords) | ⚠️ Loanwords only | High (Conversational & extempore speech) | **CC-BY-4.0** <br> ⚠️ Gated (`gated: auto`, requires HF token) | **Eval & Training** (Stream 20–30 samples for Eval) | **SUITABLE (EVAL & TRAIN - GATED)** |
| **CoSTA Test Split** (`csalt-research/CoSTA`) | ✅ **YES** (Human code-switched test recordings) | ✅ **YES** (Te-En pair) | ✅ **YES** (Genuine intra-sentential switching) | ✅ **YES** (Embedded in speech) | Very High (Academic & conversational code-mixing) | **Academic Open Access** (GitHub repo / tarball) | **Evaluation Only** (Training split is synthetic) | **SUITABLE FOR EVAL ONLY (TEST SPLIT)** |
| **AI4Bharat Kathbath** (`ai4bharat/IndicSuperb`) | ✅ **YES** (12 AP/TS regional dialect districts) | ✅ **YES** (~100 hrs) | ❌ **LOW** (Read vernacular speech) | ❌ **NO** | High (Dialectal accent & regional robustness) | **CC-BY-4.0** <br> ⚠️ Gated (`gated: auto`, requires HF token) | **Eval & Training** (Dialectal robustness) | **SUITABLE (REGIONAL EVAL & TRAIN - GATED)** |
| **Google FLEURS** (`google/fleurs`) | ✅ **YES** (Crowdsourced native speakers, 16kHz) | ✅ **YES** (12.5 hrs `te_in`) | ❌ **NO** (Monolingual read speech) | ✅ **YES** (12.0 hrs `en_us`) | Moderate (General domain / Wikipedia text) | **CC-BY-4.0** (Open access, no gating) | **Evaluation Only** (Baseline & English regression) | **SUITABLE (STREAMING EVALUATION)** |
| **LibriSpeech Clean** (`librispeech_asr`) | ✅ **YES** (High-quality studio audiobook speech) | ❌ **NO** | ❌ **NO** | ✅ **YES** (100% clean English) | High (Standard English benchmark baseline) | **CC-BY-4.0** (Open access) | **Evaluation Only** (English regression check) | **SUITABLE (ENGLISH REGRESSION EVAL)** |
| **AI4Bharat Vistaar** (`ai4bharat/vistaar`) | ✅ **YES** (Multi-domain field, news & lectures) | ✅ **YES** (Multi-benchmark) | ⚠️ **MODERATE** (Lectures contain loanwords) | ❌ **NO** | High (Multi-sector acoustic robustness) | **CC-BY-4.0** (Public benchmarks) | **Evaluation Only** (Multi-domain benchmark) | **SUITABLE (MULTI-DOMAIN EVAL)** |
| **Local Studio Corpus** (`dataset/telugu_tts/raw_*`) | ✅ **YES** (140 Studio WAVs, IISc & IIT-M) | ✅ **YES** (Clean studio speech) | ❌ **LOW** | ❌ **NO** | Moderate (Clean acoustic validation) | **Local Disk** | **Local Sanity Only** <br> 🚫 **QUARANTINED FROM ASR TRAINING** | **QUARANTINED (NO ASR TRAINING)** |
| **Production Benchmark Set** (`benchmark_test_set/`) | ⚠️ **2 Real, 13 macOS TTS** | ✅ **YES** | ✅ **YES** | ✅ **YES** | Very High (Our exact academic queries) | **Local Quarantined** | **Frozen Production Regression Test** | **FROZEN REGRESSION SET** |

---

## 2. In-Depth Reality Check Analysis

### 2.1 Why Pure IndicVoices / Kathbath Alone Is Not Enough
While **IndicVoices** and **Kathbath** are the gold standards for natural Telugu phonology across regional dialects, they were recorded as general conversations and read news. They have limited density of specific higher-education administrative keywords (*Qdrant, SGPA, condonation fee, revaluation deadline*).

### 2.2 Why CoSTA Is Essential for Code-Mixing Evaluation
The **CoSTA Test Benchmark** (IIT Bombay, COLING 2025) is the only openly verified public dataset providing **genuine human-recorded intra-sentential Telugu-English code-switching** with bilingual transcripts. However, only the **test split** contains real human recordings; the training split contains synthetic alignments and must NOT be used for ASR ground truth.

### 2.3 The Quarantine of Local Studio Data (`dataset/telugu_tts/`)
Inspection revealed that `syspin_f_0002.wav` and `indictts_f_0011.wav` in `dataset/telugu_tts/` are identical to benchmark samples `A1_te_human_syspin.wav` and `A2_te_human_indictts.wav`. Training any ASR model on `dataset/telugu_tts/` would cause immediate train-test leakage into the 15-sample production benchmark.

### 2.4 Streaming & Download Strategy in Google Colab
To prevent multi-gigabyte disk bloat and network timeouts:
1. **Google FLEURS (`te_in` / `en_us`)**: Stream test splits using Hugging Face `load_dataset("google/fleurs", "te_in", split="test", streaming=True)`.
2. **AI4Bharat IndicVoices**: Stream directly from Parquet shards using `load_dataset("ai4bharat/IndicVoices", "te", split="test", streaming=True)` with `token=HF_TOKEN`.
3. **LibriSpeech Clean**: Stream using `load_dataset("librispeech_asr", "clean", split="test", streaming=True)`.
4. **CoSTA Test**: Clone the GitHub test subset directly (~450MB).

---

## 3. Verified Benchmark Allocation Target (Total ~90–110 Real Audio Samples)

| Evaluation Category | Target Sample Count | Primary Real Human Audio Source | Verification Criteria |
| :--- | :--- | :--- | :--- |
| **1. Pure Telugu (General/Academic)** | **25 Samples** | Google FLEURS `te_in` (15) + IndicVoices Read (10) | Native speakers, clean 16kHz audio, zero Devanagari bleed |
| **2. Conversational Telugu** | **15 Samples** | IndicVoices Extempore/Conversational (`te`) | Spontaneous natural speech, multi-speaker, conversational cadence |
| **3. Code-Mixed Telugu-English** | **25 Samples** | CoSTA Real Test Split (15) + IndicVoices Spontaneous (10) | Intra-sentential switching, loanword preservation (*exams, attendance*) |
| **4. Pure English (Academic/Policy)** | **20 Samples** | Google FLEURS `en_us` (10) + LibriSpeech Clean (10) | 100% English acoustic baseline, zero English regression |
| **5. Technical / IT Terminology** | **10 Samples** | Curated real human recordings / IndicVoices Tech | High density of technical entities (*Wi-Fi, database, system*) |
| **6. Short Inquiries (<2.5s)** | **10 Samples** | Short conversational clips from IndicVoices / FLEURS | Low-latency rapid command recognition |
| **7. Long Multi-Clause Inquiries (>5.0s)** | **10 Samples** | Multi-clause sentences from CoSTA / IndicVoices | Long sequence stability, zero repetition loops |
| **Total Benchmark Footprint** | **~115 Samples** | **100% Real Human Audio** | **Zero leakage against quarantined 15 benchmark files** |
