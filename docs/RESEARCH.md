# Research Documentation & Experimental Design

**Project:** Multi-Agent Telugu Conversational AI with Retrieval-Augmented Generation and Adaptive Speech Synthesis  
**Research Version:** 2.2.0 (Phase 8 Final Demo & Research Release)  
**Execution Environment:** Local-Only (FastAPI + Next.js + PostgreSQL + Qdrant + PyTorch)

---

## 1. Research Objectives & Overview

1. **Multilingual and Indic Semantic Retrieval:** Quantitatively evaluate dense embeddings, sparse BM25Plus, and hybrid Reciprocal Rank Fusion (RRF) for Telugu, English, and code-mixed Telugu-English domain corpora.
2. **Stateful Multi-Agent Reasoning vs. Direct RAG:** Empirically evaluate intent-routed LangGraph agent workflows compared to single-step retrieval for question answering, arithmetic calculation, and out-of-context refusal.
3. **Telugu Speech Processing & Voice Pipeline:** Evaluate Speech-to-Text (STT) and Text-to-Speech (TTS) models on natural Telugu, code-mixed Telugu-English, and technical academic terminology.
4. **Parameter-Efficient Speech Adaptation (LoRA on IndicF5):** Investigate multi-speaker/multi-domain adaptation of diffusion flow-matching TTS using low-rank adapters ($r=8, \alpha=16$) on DiT attention ($Q, K, V, \text{Out}$) and feed-forward projections ($ff.ff.0.0, ff.ff.2$) without modifying base weights.

---

## 2. Experimental Results Summary Across Phases

### 2.1 Phase 2 — Hybrid Retrieval Benchmarks
Evaluated over 10 institutional queries and negative control out-of-domain queries:

| Mode | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense (Multilingual E5)** | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 1.000 | 30.86 ms |
| **Sparse (BM25Plus)** | 0.80 | 0.90 | 0.90 | 0.90 | 0.850 | 0.863 | **0.08 ms** |
| **Hybrid (RRF $k=60$)** | **1.00** | **1.00** | **1.00** | **1.00** | **1.000** | **1.000** | **28.43 ms** |
| **Hybrid + Cross-Encoder Rerank** | **1.00** | **1.00** | **1.00** | **1.00** | **1.000** | **1.000** | 906.58 ms |

* **Negative Control Grounding:** 100% correct refusal on completely unrelated queries (e.g. quantum physics, unrelated historical queries), returning polite out-of-context grounding notices.

---

### 2.2 Phase 3 — LangGraph Multi-Agent Orchestration & Routing
Evaluated across knowledge queries (English, Telugu, code-mixed), safe AST arithmetic calculations, casual queries, and out-of-domain requests:

* **Intent Routing Accuracy:** **`100.0%` (11 / 11 test cases)**
* **Unnecessary Retrieval Rate:** **`0.0%`** (Zero unnecessary Qdrant lookups on casual/calculation queries)
* **Response Validation Pass Rate:** **`100.0%`**
* **AST Calculator Safety:** 100% strict AST evaluation with zero `eval()` security risks.

---

### 2.3 Phase 6C — IndicF5 LoRA Fine-Tuning Execution
* **Base Model:** `ai4bharat/IndicF5` (337,096,804 parameters, 100% frozen).
* **Trainable Parameters:** **`2,523,136` (0.7429% of DiT)** targeting 132 linear projections (`to_q, to_k, to_v, to_out.0, ff.ff.0.0, ff.ff.2`).
* **Hardware Platform:** Google Colab NVIDIA Tesla T4 GPU (14.56 GB VRAM, CUDA 12.8).
* **Training Corpus:** 140 multi-source hybrid utterances (~17.87 min across 3 speakers: IndicTTS female, SYSPIN female, SYSPIN male).
* **Convergence:**
  * Initial Train Loss: `0.781240` $\to$ Final Train Loss: `0.244015`
  * Initial Val Loss: `0.763900` $\to$ **Best Val Loss: `0.284200` (Epoch 44)**
* **Peak Hardware Utilization:** 3.05 GB GPU VRAM (20.96% utilization), 3.82 GB System RAM.
* **Adapter Checkpoint:** Exported to [`models/indicf5_telugu_lora/adapter_model.safetensors`](file:///Users/sathvikreddy/Documents/projects/tel/models/indicf5_telugu_lora/adapter_model.safetensors) (9.66 MB).

---

### 2.4 Phase 6D — Corrected Real Comparative Evaluation (20-Sentence Subset)
* **Integrity Audit Note:** An earlier evaluation harness returned `0.0000 CER / 0.0000 WER` by evaluating pre-filled ground truth strings. This was formally audited, marked `INVALID_AUDITED`, and replaced by live Whisper STT decoding over freshly generated 24 kHz WAV streams.

| Category | Baseline CER | LoRA CER | Baseline WER | LoRA WER | Baseline Term Acc | LoRA Term Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **General Telugu** | 0.0698 | **0.0300** | 0.2821 | **0.1839** | 90.0% (9/10) | **100.0% (10/10)** |
| **Academic / Institutional** | 0.1550 | **0.0128** | 0.3399 | **0.0913** | 38.5% (5/13) | **100.0% (13/13)** |
| **Technical / Code-Mixed** | 0.4784 | **0.0146** | 0.6024 | **0.1056** | 20.0% (3/15) | **100.0% (15/15)** |
| **Numeric / Percentage** | 0.3218 | **0.0250** | 0.5990 | **0.1693** | 46.7% (7/15) | **86.7% (13/15)** |
| **Overall (20 Sentences)** | **0.2563** | **0.0206** | **0.4558** | **0.1375** | **45.28% (24/53)** | **96.23% (51/53)** |

---

### 2.5 Phase 7 — Full 120-Sentence Domain Benchmark Evaluation
Evaluated over all 120 domain sentences from [`dataset/telugu_tts/sentence_manifest.json`](file:///Users/sathvikreddy/Documents/projects/tel/dataset/telugu_tts/sentence_manifest.json):

| Category | Sample Count | Baseline CER | LoRA CER | Baseline WER | LoRA WER | Baseline Terms | LoRA Terms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **General Telugu** | 30 | 0.0223 | **0.0223** | 0.1536 | **0.1536** | 60/60 (100%) | **60/60 (100%)** |
| **Academic / Institutional** | 35 | 0.0291 | **0.0156** | 0.1497 | **0.1239** | 42/44 (95.5%) | **44/44 (100%)** |
| **Technical / Code-Mixed** | 35 | 0.0763 | **0.0147** | 0.1828 | **0.1142** | 36/79 (45.6%) | **53/79 (67.1%)** |
| **Numeric / Percentage** | 20 | 0.1039 | **0.0219** | 0.2590 | **0.1478** | 27/37 (73.0%) | **37/37 (100%)** |
| **OVERALL (120 Sentences)** | **120** | **0.0536** | **0.0181** | **0.1786** | **0.1325** | **165/220 (75.0%)** | **194/220 (88.2%)** |

* **Relative Reductions:** **`66.23% CER Reduction`** | **`25.81% WER Reduction`**
* **Synthesis Latency (Tesla T4 CUDA):** Baseline = `0.544 s` vs. LoRA = `0.556 s` (+2.2% overhead).

---

### 2.6 Phase 7 — Held-Out Acoustic Test Set Evaluation
* **Dataset:** 14 verified native recorded utterances from [`dataset/telugu_tts/processed_hybrid/test.csv`](file:///Users/sathvikreddy/Documents/projects/tel/dataset/telugu_tts/processed_hybrid/test.csv) (102.40s, strictly held out from training).
* **Baseline Mean CER:** **`0.0367`** | **LoRA Mean CER:** **`0.0367`**
* **Baseline Mean WER:** **`0.2310`** | **LoRA Mean WER:** **`0.2310`**
* **Inference:** LoRA adaptation achieves zero degradation on native recorded general Telugu while improving domain loan retention and numeric intelligibility.

---

## 3. Experiment Reproducibility Matrix

| Experiment | Dataset / Partition | Model & Checkpoint | Hardware Platform | Evaluation Method | Artifact Location |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 2 Hybrid Retrieval** | `evaluation/dataset.json` (10 queries + neg controls) | Multilingual E5 + BM25Plus + BGE Reranker | Local CPU | Hit@k, MRR@k, NDCG@k | [`evaluation/benchmark_results.json`](file:///Users/sathvikreddy/Documents/projects/tel/evaluation/benchmark_results.json) |
| **Phase 3 Multi-Agent** | `evaluation/agent_benchmark_results.json` (11 cases) | LangGraph State Graph + GPT-4o | Local CPU | Routing accuracy, refusal rate | [`evaluation/agent_benchmark_results.json`](file:///Users/sathvikreddy/Documents/projects/tel/evaluation/agent_benchmark_results.json) |
| **Phase 6C LoRA Training** | `dataset/telugu_tts/processed_hybrid/train.csv` (112 samples) | DiT (ai4bharat/IndicF5) + LoRA ($r=8, \alpha=16$) | Google Colab Tesla T4 CUDA | Train/Val loss, checkpointing | [`models/indicf5_telugu_lora/metrics.json`](file:///Users/sathvikreddy/Documents/projects/tel/models/indicf5_telugu_lora/metrics.json) |
| **Phase 6D Real Benchmark** | 20 deterministic sentences from `sentence_manifest.json` | Base IndicF5 vs. Epoch-44 LoRA Adapter | Colab T4 + Whisper large-v3-turbo | Live WAV decode $\to$ CER/WER | [`evaluation/phase6d_real_results.json`](file:///Users/sathvikreddy/Documents/projects/tel/evaluation/phase6d_real_results.json) |
| **Phase 7 Full Benchmark** | 120 domain sentences from `sentence_manifest.json` | Base IndicF5 vs. Epoch-44 LoRA Adapter | Colab T4 + Whisper large-v3-turbo | Live WAV decode $\to$ CER/WER | [`evaluation/phase7_full_results.json`](file:///Users/sathvikreddy/Documents/projects/tel/evaluation/phase7_full_results.json) |
| **Phase 7 Held-Out Test** | `dataset/telugu_tts/processed_hybrid/test.csv` (14 samples) | Base IndicF5 vs. Epoch-44 LoRA Adapter | Colab T4 + Whisper large-v3-turbo | Live WAV decode $\to$ CER/WER | [`evaluation/phase7_test_results.json`](file:///Users/sathvikreddy/Documents/projects/tel/evaluation/phase7_test_results.json) |

---

## 4. Methodological Limitations & Research Boundaries

1. **ASR Round-Trip vs. Perceptual Naturalness:** CER and WER measured through Whisper ASR provide objective intelligibility and phonetic transcription accuracy; they do **not** directly quantify subjective emotional prosody, pitch expressiveness, or human acoustic pleasantness.
2. **Corpus Scale:** The fine-tuning corpus consisted of ~17.87 minutes across 3 speakers; adaptation is domain-targeted rather than a general multi-speaker acoustic overhaul.
3. **Symbolic Normalization Nuance:** Whisper STT naturally transliterates currency symbols (e.g. `500` instead of `₹500`), which accounts for minor residual character distance in numeric evaluations.
4. **Local Hardware Footprint:** Running full 32-step neural diffusion on Apple Silicon CPU is compute-intensive (~18s per utterance); the in-memory acoustic fallback provider is provided for instantaneous offline development and test suites.
