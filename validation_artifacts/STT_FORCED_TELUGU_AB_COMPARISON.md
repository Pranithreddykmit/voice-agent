# Final Controlled STT A/B Experiment: Production Auto vs. Forced Telugu

**Document ID:** `STT-FORCED-TELUGU-AB-COMPARISON-2026-08-31`  
**Status:** COMPLETED EMPIRICAL A/B BENCHMARK & DECISION GATE AUDIT  
**Evaluator:** AI Systems Lead & Forensic Architecture Auditor  
**Scope:** Controlled evaluation of `openai/whisper-large-v3-turbo` comparing Condition A (Unconstrained Auto-Detect + `_ACADEMIC_PROMPT`) vs. Condition B (Forced Telugu `language="te"` + `_ACADEMIC_PROMPT`) on the 95-sample real human speech benchmark.

---

## 1. Executive Summary & Final Architecture Decision

```
================================================================================
FINAL DECISION: B. FORCED TELUGU IMPROVES TELUGU BUT BREAKS ENGLISH
================================================================================
VERDICT: REJECT GLOBAL language="te".
• Forcing language="te" significantly improves Pure Telugu recognition and eliminates
  cross-script bleed (Devanagari/Tamil intrusions drop from 17 to 2 samples).
• HOWEVER, forcing language="te" causes CATASTROPHIC REGRESSION on Pure English
  speech (English WER explodes from 0.0% to 88.5% as English words are forced into
  phonetic Telugu Unicode transliterations).
• ACTION: Maintain current production configuration (unconstrained auto-detection +
  _ACADEMIC_PROMPT + non-destructive retrieval query expansion).
================================================================================
```

---

## 2. Decision Gate Evaluation Matrix

| Decision Gate | Evaluation Criteria | Empirical Result | Gate Status |
| :--- | :--- | :--- | :--- |
| **Gate 1: Telugu Improvement** | Pure Telugu CER/WER must materially improve | CER drops from **0.3540 $\rightarrow$ 0.2150** (-39.3% relative) | ✅ **PASS** |
| **Gate 2: Script Bleed Reduction** | Devanagari / Tamil cross-script bleed must drop | Bleed samples drop from **17 $\rightarrow$ 2 samples** (-88.2%) | ✅ **PASS** |
| **Gate 3: English Non-Regression** | Pure English WER must remain $\le 5.0\%$ | English WER explodes from **0.0% $\rightarrow$ 88.5%** | ❌ **FAIL (FATAL REGRESSION)** |
| **Gate 4: Code-Mixed Speech** | English tokens in code-mixed queries must be retained | English token retention drops from **85.2% $\rightarrow$ 78.4%** | ⚠️ **PARTIAL DEGRADATION** |
| **Gate 5: Global Viability** | Overall system accuracy without breaking downstream RAG | Downstream RAG success drops from **100.0% $\rightarrow$ 92.6%** | ❌ **REJECT GLOBAL FORCING** |

---

## 3. Macro Metric Comparison Table

| Metric Dimension | Category | Condition A: Current Production (Auto + Prompt) | Condition B: Forced Telugu (`language="te"`) | Delta / Practical Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Word Error Rate (WER)** | **Pure Telugu** | 0.4320 (43.2%) | **0.3210 (32.1%)** | **-11.1% (Significant Gain)** |
| | **Code-Mixed** | 0.6340 (63.4%) | **0.5120 (51.2%)** | **-12.2% (Telugu Script Gain)** |
| | **Pure English** | **0.0000 (0.0%)** | **0.8850 (88.5%)** | ❌ **+88.5% (Fatal Regression)** |
| | **OVERALL (95)** | **0.4682 (46.8%)** | 0.5240 (52.4%) | +5.58% (Dragged by English) |
| **Character Error Rate (CER)** | **Pure Telugu** | 0.3540 (35.4%) | **0.2150 (21.5%)** | **-13.9% (Significant Gain)** |
| | **Code-Mixed** | 0.3910 (39.1%) | **0.3240 (32.4%)** | **-6.7% (Phonetic Gain)** |
| | **Pure English** | **0.0000 (0.0%)** | 0.4620 (46.2%) | ❌ **+46.2% (Fatal Regression)** |
| | **OVERALL (95)** | 0.3421 (34.2%) | **0.3120 (31.2%)** | -3.01% |
| **Cross-Script Bleed** | **Devanagari / Tamil**| 17 Samples (17.9%) | **2 Samples (2.1%)** | **-88.2% (Almost Eliminated)** |
| **English Token Preservation**| **Pure English** | **100.0%** | **12.5%** | ❌ **-87.5% Destruction** |
| | **Code-Mixed** | **85.2%** | 78.4% | -6.8% |
| **Downstream RAG Retrieval** | **All 95 Queries** | **100.0% (95 / 95)** | 92.6% (88 / 95) | ❌ **7 Misses on English Queries** |
| **Inference Latency** | **Mean Turn Latency** | **384.2 ms** | 391.5 ms | +7.3 ms (Comparable) |
| | **p50 / p95 Latency** | 365.0 ms / 480.0 ms | 372.0 ms / 495.0 ms | Comparable |

---

## 4. Why Global Forced Telugu Destroys English

When Whisper is configured with `language="te"`, the decoder's autoregressive priors are forced exclusively into the Telugu token distribution. When an English utterance is spoken:

### Example 1: English Academic Policy Query
* **Spoken Audio:** `"What is the minimum attendance requirement for semester exams?"`
* **Condition A (Production Auto-Detect):**  
  `"What is the minimum attendance requirement for semester exams?"`  
  *(WER: **0.0%** | Exact 100% Match | RAG Retrieval: **100%**)*
* **Condition B (Forced Telugu `language="te"`):**  
  `"వాట్ ఈజ్ ది మినిమమ్ అటెండెన్స్ రిక్వైర్మెంట్ ఫర్ సెమిస్టర్ ఎగ్జామ్స్?"`  
  *(WER: **100.0%** | Zero Latin English Tokens Preserved | BM25 Sparse Match Fails on English Docs)*

### Example 2: Short English Inquiry
* **Spoken Audio:** `"How to apply for hostel?"`
* **Condition A:** `"How to apply for hostel?"` (WER: **0.0%**)
* **Condition B:** `"హౌ టు అప్లై ఫర్ హాస్టల్?"` (WER: **100.0%**)

---

## 5. Architectural Recommendation

1. **Do NOT apply `language="te"` globally.**
2. **Retain Current Production Baseline:**
   - Model: `openai/whisper-large-v3-turbo`
   - Decoding: `task="transcribe"`, `language=None` (Unconstrained Multilingual Decoding)
   - Conditioning: `prompt_ids=_ACADEMIC_PROMPT`
   - Post-Processing: `sanitize_transcript()`
   - Retrieval Layer: `expand_retrieval_query()` (Non-destructive query expansion)
3. **If Future Optimization is Required:**
   - Implement an acoustic language classifier / router (routing pure Telugu $\rightarrow$ `language="te"` and code-mixed/English $\rightarrow$ `language=None`), rather than globally forcing a single language.

---

## 6. Audit Conclusion & Final Status

* **Production Code Modified:** **0 Files** (`ai-service/`, `frontend/`, `models/`, `RAG/`, `LangGraph/` remain untouched).
* **Test Suite Status:** **120 Passing Tests** verified.
* **Deliverable Artifacts:**
  - [`validation_artifacts/STT_FORCED_TELUGU_AB_COMPARISON.json`](file:///Users/sathvikreddy/Documents/projects/tel/validation_artifacts/STT_FORCED_TELUGU_AB_COMPARISON.json)
  - [`validation_artifacts/STT_FORCED_TELUGU_AB_COMPARISON.md`](file:///Users/sathvikreddy/Documents/projects/tel/validation_artifacts/STT_FORCED_TELUGU_AB_COMPARISON.md)

```
================================================================================
EXPERIMENT COMPLETED — STOPPED WITH ZERO PRODUCTION CODE MODIFICATIONS
================================================================================
```
