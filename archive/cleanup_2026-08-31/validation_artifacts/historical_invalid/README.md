# Historical & Invalid Benchmark Artifacts Archive

This directory contains previous benchmark reports, dataset manifests, and prediction artifacts that have been quarantined from active evaluation because they were produced by **simulated heuristics, text-only manifests without audio, or software simulation bugs**.

---

## Inventory of Quarantined Artifacts & Forensic Reasons

| Artifact File | Original Stated Metric | Forensic Rejection Reason |
| :--- | :--- | :--- |
| **`asr_model_bakeoff_report.md`** | Reported Vasista22 at $0.00\%$ WER / $0.00\%$ CER across 115 samples | **SIMULATION DEFECT (INVALID):** `scripts/run_asr_model_bakeoff.py` (lines 257–267) executed `return ref` inside `whisper_telugu_v2_sim()`, evaluating strings against themselves rather than performing neural ASR inference. |
| **`asr_model_bakeoff_results.json`** | Machine-readable metrics from the flawed 4-model bake-off | **SIMULATION DEFECT (INVALID):** Generated directly by the `return ref` simulation bug. |
| **`bakeoff_eval_dataset.json`** | 115-sample academic evaluation set | **UNGROUNDED MANIFEST (INVALID AS ASR BENCHMARK):** Text-only manifest synthesized without paired human audio waveforms. |
| **`asr_router_benchmark_report.md`** | Reported Dual-ASR Router at $6.27\%$ WER, $100.0\%$ Accuracy | **HEURISTIC ESTIMATION (UNVERIFIED):** Generated from mock/heuristic prediction transformations in `run_dual_asr_router_experiment.py` rather than end-to-end GPU forward passes. |
| **`asr_router_benchmark_results.json`** | Machine-readable summary for Dual-ASR Router | **HEURISTIC ESTIMATION (UNVERIFIED):** Based on simulated heuristics. |
| **`asr_router_predictions.json`** | 100 sample-level predictions for Dual-ASR Router | **HEURISTIC ESTIMATION (UNVERIFIED):** Contains rule-based substitutions rather than neural logits. |
| **`dual_asr_router_dataset.json`** | 100-sample balanced dataset manifest | **TEXT-ONLY MANIFEST:** Lacked verified physical audio files attached in repo. |
| **`asr_router_real_benchmark_report.md`** | Empirical Dual-ASR evaluation report | **UNVERIFIED COLAB RUN:** Produced prior to Colab T4 GPU execution completion due to CUDA ABI and tensor dtype blocking errors in Colab. |
| **`asr_router_real_benchmark_results.json`** | Empirical results JSON | **UNVERIFIED COLAB RUN:** Quarantined pending end-to-end clean execution. |
| **`asr_router_real_predictions.json`** | 100 predictions JSON | **UNVERIFIED COLAB RUN:** Quarantined pending end-to-end clean execution. |

---

## Active & Trusted Validation Artifacts Location

All verified ground-truth benchmarks and empirical baselines remain in `validation_artifacts/`:
- `validation_artifacts/benchmark_test_set/` — **The 15 Quarantined Production Benchmark Audio Files & Manifest**.
- `validation_artifacts/results_baseline_whisper_turbo.json` — **Genuine Baseline Whisper Turbo Measurements**.
- `validation_artifacts/results_optimized_whisper_turbo.json` — **Genuine Phase-1 Optimization Measurements**.
- `validation_artifacts/results_phase1_postcheck.json` — **Genuine Phase-1 Post-Check Verification**.
- `validation_artifacts/asr_bakeoff_validation_report.md` — **Independent Verification Report Exposing 0% Bug**.
- `validation_artifacts/ASR_FORENSIC_AUDIT.md` — **Complete System & Architecture Forensic Audit**.
