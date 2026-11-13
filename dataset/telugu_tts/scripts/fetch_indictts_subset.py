"""
Deterministic Subset Selection & Fetcher for IndicTTS Telugu (Phase 6B.5).
Streams female speaker utterances from SPRINGLab/IndicTTS_Telugu, filters by duration
and audio quality, and saves reproducible manifest and raw audio.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset.fetch_indictts")

TARGET_SAMPLE_COUNT = 60
MIN_DUR = 1.0
MAX_DUR = 20.0


def fetch_indictts_female_subset(
    output_dir: str = "dataset/telugu_tts/raw_indictts",
    manifest_out: str = "dataset/telugu_tts/indictts_selected_manifest.json",
    target_count: int = TARGET_SAMPLE_COUNT
) -> Dict[str, Any]:
    """Streams and selects a clean, reproducible subset of female IndicTTS Telugu utterances."""
    from datasets import load_dataset

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Connecting to SPRINGLab/IndicTTS_Telugu streaming dataset...")
    ds = load_dataset("SPRINGLab/IndicTTS_Telugu", split="train", streaming=True)

    selected = []
    seen_texts = set()

    # Stratified duration bins: 20 short (1-4s), 25 medium (4-10s), 15 long (10-20s)
    short_samples = []
    med_samples = []
    long_samples = []

    target_short = int(target_count * 0.35)
    target_med = int(target_count * 0.40)
    target_long = target_count - target_short - target_med

    logger.info(f"Target distribution: {target_short} short, {target_med} medium, {target_long} long utterances")

    for idx, item in enumerate(ds):
        if len(selected) >= target_count:
            break

        # Filter strictly for female speaker (gender == 0)
        if item["gender"] != 0:
            continue

        text = item["text"].strip()
        if not text or text in seen_texts:
            continue

        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        duration = len(audio_array) / float(sr)

        if duration < MIN_DUR or duration > MAX_DUR:
            continue

        peak = np.max(np.abs(audio_array))
        if peak > 0.98 or peak < 0.05:
            continue

        seen_texts.add(text)
        item_data = {
            "source_idx": idx,
            "text": text,
            "audio_array": audio_array,
            "orig_sr": sr,
            "duration": round(duration, 2),
            "peak": round(float(peak), 3)
        }

        if duration <= 4.0 and len(short_samples) < target_short:
            short_samples.append(item_data)
        elif 4.0 < duration <= 10.0 and len(med_samples) < target_med:
            med_samples.append(item_data)
        elif 10.0 < duration <= MAX_DUR and len(long_samples) < target_long:
            long_samples.append(item_data)

        if len(short_samples) >= target_short and len(med_samples) >= target_med and len(long_samples) >= target_long:
            break

    all_candidates = short_samples + med_samples + long_samples
    manifest_records = []

    for i, item in enumerate(all_candidates, start=1):
        sample_id = f"indictts_f_{i:04d}"
        wav_file = out_path / f"{sample_id}.wav"
        sf.write(str(wav_file), item["audio_array"], item["orig_sr"], subtype="PCM_16")

        manifest_records.append({
            "id": sample_id,
            "source": "indictts",
            "speaker": "indic_female",
            "category": "general",
            "text": item["text"],
            "raw_wav": str(wav_file.resolve()),
            "duration_sec": item["duration"],
            "orig_sample_rate": item["orig_sr"]
        })

    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    total_dur = sum(r["duration_sec"] for r in manifest_records)
    summary = {
        "selected_samples": len(manifest_records),
        "female_speaker": "indic_female",
        "total_duration_sec": round(total_dur, 2),
        "total_duration_min": round(total_dur / 60.0, 2),
        "short_count": len(short_samples),
        "med_count": len(med_samples),
        "long_count": len(long_samples),
        "manifest_path": manifest_out,
        "raw_dir": str(out_path)
    }

    logger.info("=" * 60)
    logger.info("INDICTTS FEMALE SUBSET SELECTION COMPLETED")
    logger.info(f"Selected:       {summary['selected_samples']} utterances")
    logger.info(f"Total Duration: {summary['total_duration_sec']}s ({summary['total_duration_min']} min)")
    logger.info(f"Bins:           {summary['short_count']} short, {summary['med_count']} med, {summary['long_count']} long")
    logger.info(f"Manifest:       {summary['manifest_path']}")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch IndicTTS Female Subset")
    parser.add_argument("--count", type=int, default=TARGET_SAMPLE_COUNT, help="Number of utterances to select")
    parser.add_argument("--output-dir", type=str, default="dataset/telugu_tts/raw_indictts")
    parser.add_argument("--manifest-out", type=str, default="dataset/telugu_tts/indictts_selected_manifest.json")
    args = parser.parse_args()

    fetch_indictts_female_subset(
        output_dir=args.output_dir,
        manifest_out=args.manifest_out,
        target_count=args.count
    )
