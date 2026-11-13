"""
Phase-1 STT Optimisation — Post-Implementation Benchmark
=========================================================
Loads the production WhisperSTTProvider from ai-service (with Phase-1 changes active)
and runs it against the 15-sample benchmark manifest used in the baseline measurement.

Changes active in this run:
  ✅  repetition_penalty removed
  ✅  no_repeat_ngram_size removed
  ✅  _probe_target_language removed
  ✅  Bilingual academic prompt_ids conditioning active
  ✅  Native Whisper multilingual language detection
  ✅  Device: auto (MPS on Apple Silicon M4)

Usage:
    cd /Users/sathvikreddy/Documents/projects/tel
    .venv/bin/python scripts/run_asr_phase1_postcheck.py

Output:
    validation_artifacts/results_phase1_postcheck.json
"""

import sys
import gc
import json
import time
import psutil
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# ── repo paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ai-service"))

MANIFEST_PATH = REPO_ROOT / "validation_artifacts" / "benchmark_test_set" / "manifest.json"
BASELINE_PATH = REPO_ROOT / "validation_artifacts" / "results_baseline_whisper_turbo.json"
OUT_PATH      = REPO_ROOT / "validation_artifacts" / "results_phase1_postcheck.json"

# ── metrics helpers ──────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.lower().strip())


def compute_wer(ref: str, hyp: str) -> float:
    r = _normalise(ref).split()
    h = _normalise(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return round(d[len(r)][len(h)] / len(r), 4)


def compute_cer(ref: str, hyp: str) -> float:
    r = _normalise(ref).replace(" ", "")
    h = _normalise(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return round(d[len(r)][len(h)] / len(r), 4)


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= c <= "\u097F" for c in text)


def categorize_errors(ref: str, hyp: str) -> list:
    tags = []
    if has_devanagari(hyp) and not has_devanagari(ref):
        tags.append("Devanagari script bleed")
    if not tags:
        wer = compute_wer(ref, hyp)
        if wer == 0.0:
            tags.append("Clean / Faithful")
        else:
            tags.append("Phonetic/Subword mismatch")
    return tags


def get_mem() -> dict:
    proc = psutil.Process()
    rss_mb = round(proc.memory_info().rss / 1024 / 1024, 2)
    vm = psutil.virtual_memory()
    sys_used_gb = round((vm.total - vm.available) / 1024 ** 3, 2)
    return {"rss_mb": rss_mb, "sys_used_gb": sys_used_gb}


# ── main benchmark ───────────────────────────────────────────────────────────
def run_phase1_benchmark():
    print("\n" + "=" * 80)
    print("PHASE-1 STT OPTIMISATION — POST-IMPLEMENTATION BENCHMARK")
    print("Changes: no repetition_penalty | no probe | bilingual prompt | MPS auto")
    print("=" * 80)

    mem_before = get_mem()
    print(f"\nMemory Before Load: RSS={mem_before['rss_mb']}MB | System={mem_before['sys_used_gb']}GB")

    # ── load production provider ─────────────────────────────────────────────
    # This imports the real WhisperSTTProvider with all Phase-1 changes active.
    from app.speech.stt import WhisperSTTProvider
    t_load = time.perf_counter()
    provider = WhisperSTTProvider()            # uses settings.STT_MODEL_NAME + settings.STT_DEVICE
    load_ms = (time.perf_counter() - t_load) * 1000.0
    mem_loaded = get_mem()
    print(f"Model Loaded in {load_ms:.0f}ms | RSS={mem_loaded['rss_mb']}MB "
          f"(+{mem_loaded['rss_mb'] - mem_before['rss_mb']:.0f}MB) | "
          f"Device={provider._torch_device}")
    print()

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    for sample in manifest:
        data, sr = sf.read(sample["wav_path"])
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        t0 = time.perf_counter()
        transcript, detected_lang, lat_ms = provider.transcribe(data, sample_rate=sr)
        # lat_ms is already measured inside provider; re-compute wall time for consistency
        wall_ms = (time.perf_counter() - t0) * 1000.0

        dur_sec = sample["duration_sec"]
        rtf = round(wall_ms / 1000.0 / dur_sec, 3) if dur_sec > 0 else 0.0
        wer = compute_wer(sample["ref_text"], transcript)
        cer = compute_cer(sample["ref_text"], transcript)
        err_tags = categorize_errors(sample["ref_text"], transcript)

        entry = {
            "id": sample["id"],
            "category": sample["category"],
            "duration_sec": dur_sec,
            "latency_ms": round(wall_ms, 2),
            "rtf": rtf,
            "detected_lang": detected_lang,
            "ref_text": sample["ref_text"],
            "transcript": transcript,
            "wer": wer,
            "cer": cer,
            "errors": err_tags,
        }
        results.append(entry)

        print(f"[{sample['id']}] ({dur_sec}s → {wall_ms:.0f}ms, RTF={rtf:.2f}x, lang={detected_lang})")
        print(f"   Ref:  {sample['ref_text']}")
        print(f"   Hyp:  {transcript}")
        print(f"   WER:  {wer*100:.1f}% | CER: {cer*100:.1f}% | Tags: {', '.join(err_tags)}\n")

    # ── aggregate stats ──────────────────────────────────────────────────────
    cat_wer: dict = {}
    cat_cer: dict = {}
    for r in results:
        cat = r["category"][0]   # 'A', 'B', 'C', etc.
        cat_wer.setdefault(cat, []).append(r["wer"])
        cat_cer.setdefault(cat, []).append(r["cer"])

    avg_wer = {k: round(sum(v) / len(v) * 100, 1) for k, v in cat_wer.items()}
    avg_cer = {k: round(sum(v) / len(v) * 100, 1) for k, v in cat_cer.items()}
    overall_latency = round(sum(r["latency_ms"] for r in results) / len(results), 1)

    print("=" * 80)
    print("AGGREGATE RESULTS (AFTER Phase-1 Changes)")
    print("=" * 80)
    for cat, wer in sorted(avg_wer.items()):
        label = {"A": "Pure Telugu", "B": "Pure English", "C": "Code-Mixed",
                 "D": "Telugu+Tech", "E": "Short Queries", "F": "Long Mixed"}.get(cat, cat)
        print(f"  Category {cat} ({label}):  avg WER={wer}%  avg CER={avg_cer.get(cat,'?')}%")
    print(f"\n  Overall Average Latency : {overall_latency} ms")

    # ── unload ───────────────────────────────────────────────────────────────
    del provider
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    mem_after = get_mem()
    print(f"\nMemory After Unload: RSS={mem_after['rss_mb']}MB | System={mem_after['sys_used_gb']}GB")

    # ── BEFORE/AFTER comparison ──────────────────────────────────────────────
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH, encoding="utf-8") as f:
            baseline = json.load(f)
        b_map = {r["id"]: r for r in baseline}
        print("\n" + "=" * 80)
        print("BEFORE → AFTER COMPARISON (key samples)")
        print("=" * 80)
        for r in results:
            bid = r["id"]
            if bid not in b_map:
                continue
            b = b_map[bid]
            delta_wer = round((r["wer"] - b["wer"]) * 100, 1)
            sign = "+" if delta_wer >= 0 else ""
            print(f"\n[{bid}]")
            print(f"  Ref:    {r['ref_text']}")
            print(f"  BEFORE: {b['transcript']}   (WER={b['wer']*100:.1f}%)")
            print(f"  AFTER:  {r['transcript']}   (WER={r['wer']*100:.1f}%)")
            print(f"  ΔWER:   {sign}{delta_wer}%  "
                  f"{'✅ IMPROVED' if delta_wer < 0 else ('✳️  SAME' if delta_wer == 0 else '⚠️  WORSE')}")

    # ── save results ─────────────────────────────────────────────────────────
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results written to: {OUT_PATH}")
    return results


if __name__ == "__main__":
    run_phase1_benchmark()
