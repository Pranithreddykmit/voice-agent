"""
Hybrid Telugu Speech Dataset Assembly, Preprocessing & Stratified Splitting (Phase 6B.6).
Combines clean IndicTTS female speech + SYSPIN male/female multi-domain speech into a
unified, zero-leakage, 24kHz mono 16-bit PCM training dataset for IndicF5 LoRA adaptation.
"""

import os
import sys
import json
import csv
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import soundfile as sf
import numpy as np

# Ensure preprocess utilities are accessible
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess import process_single_audio
from split_dataset import verify_split_leakage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset.hybrid")

TARGET_SAMPLE_RATE = 24000


def prepare_hybrid_dataset(
    indictts_manifest: str = "dataset/telugu_tts/indictts_selected_manifest.json",
    syspin_manifest: str = "dataset/telugu_tts/syspin_selected_manifest.json",
    output_dir: str = "dataset/telugu_tts/processed_hybrid",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Any]:
    """Combines, normalizes, and deterministically splits IndicTTS and SYSPIN speech data."""
    out_base = Path(output_dir)
    wavs_dir = out_base / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    indictts_path = Path(indictts_manifest)
    syspin_path = Path(syspin_manifest)

    if not indictts_path.exists():
        raise FileNotFoundError(f"IndicTTS manifest missing: {indictts_path}")
    if not syspin_path.exists():
        raise FileNotFoundError(f"SYSPIN manifest missing: {syspin_path}")

    with open(indictts_path, "r", encoding="utf-8") as f:
        indictts_data: List[Dict[str, Any]] = json.load(f)
    with open(syspin_path, "r", encoding="utf-8") as f:
        syspin_data: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(indictts_data)} IndicTTS samples and {len(syspin_data)} SYSPIN samples.")

    combined_sources = []
    for item in indictts_data:
        combined_sources.append({
            "id": item["id"],
            "source": "indictts",
            "speaker": item.get("speaker", "indic_female"),
            "category": item.get("category", "general"),
            "text": item["text"].strip(),
            "raw_wav": item["raw_wav"]
        })

    for item in syspin_data:
        combined_sources.append({
            "id": item["id"],
            "source": "syspin",
            "speaker": item.get("speaker", "syspin_speaker"),
            "category": item.get("category", "education"),
            "text": item["text"].strip(),
            "raw_wav": item["raw_wav"]
        })

    processed_records = []
    metadata_rows = []
    total_duration = 0.0

    logger.info("Normalizing all audio to 24,000 Hz mono 16-bit PCM WAV...")

    for item in combined_sources:
        sample_id = item["id"]
        raw_path = item["raw_wav"]
        text = item["text"]
        target_wav = wavs_dir / f"{sample_id}.wav"

        if not os.path.exists(raw_path):
            logger.warning(f"Raw audio missing: {raw_path}")
            continue

        success, duration, err = process_single_audio(
            raw_path=str(raw_path),
            target_path=str(target_wav),
            target_sr=TARGET_SAMPLE_RATE
        )

        if not success:
            logger.warning(f"Skipping {sample_id}: {err}")
            continue

        total_duration += duration
        metadata_rows.append({
            "audio_file": str(target_wav.resolve()),
            "text": text
        })

        processed_records.append({
            "sample_id": sample_id,
            "source": item["source"],
            "speaker": item["speaker"],
            "category": item["category"],
            "text": text,
            "audio_file": str(target_wav.resolve()),
            "audio_path": str(target_wav.resolve()),
            "duration_sec": duration,
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": 1
        })

    # Save official full metadata.csv (IndicF5 format)
    meta_csv_path = out_base / "metadata.csv"
    with open(meta_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for r in metadata_rows:
            writer.writerow([r["audio_file"], r["text"]])

    # Save research tracking manifest
    manifest_json_path = out_base / "hybrid_dataset_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=2, ensure_ascii=False)

    # Deterministic Stratified Split across Speaker and Category
    rng = random.Random(seed)
    # Group by (source, speaker)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in processed_records:
        key = f"{r['source']}::{r['speaker']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    train_split = []
    val_split = []
    test_split = []

    for key in sorted(groups.keys()):
        items = list(groups[key])
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_val - n_test

        train_split.extend(items[:n_train])
        val_split.extend(items[n_train : n_train + n_val])
        test_split.extend(items[n_train + n_val :])

    # Verify Leakage
    is_clean, violations = verify_split_leakage(train_split, val_split, test_split)
    if not is_clean:
        logger.error(f"DATASET LEAKAGE DETECTED in Hybrid Dataset: {violations}")
        raise ValueError("Hybrid dataset split failed due to data leakage!")

    # Write split CSV files
    for name, split_rows in [("train", train_split), ("val", val_split), ("test", test_split)]:
        csv_file = out_base / f"{name}.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["audio_file", "text"])
            for r in split_rows:
                writer.writerow([r["audio_path"], r["text"]])

    # Split info summary
    split_info = {
        "random_seed": seed,
        "leakage_free": is_clean,
        "counts": {
            "total": len(processed_records),
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split)
        },
        "speakers": {
            "indic_female": len([r for r in processed_records if r["speaker"] == "indic_female"]),
            "syspin_female": len([r for r in processed_records if r["speaker"] == "syspin_female"]),
            "syspin_male": len([r for r in processed_records if r["speaker"] == "syspin_male"])
        },
        "durations_sec": {
            "total": round(total_duration, 2),
            "total_min": round(total_duration / 60.0, 2),
            "train": round(sum(r["duration_sec"] for r in train_split), 2),
            "val": round(sum(r["duration_sec"] for r in val_split), 2),
            "test": round(sum(r["duration_sec"] for r in test_split), 2)
        }
    }

    with open(out_base / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    logger.info("=" * 60)
    logger.info("FINAL HYBRID TELUGU TTS DATASET PREPARED")
    logger.info(f"Total Utterances:   {split_info['counts']['total']}")
    logger.info(f"Total Duration:     {split_info['durations_sec']['total']}s ({split_info['durations_sec']['total_min']} min)")
    logger.info(f"Train / Val / Test: {split_info['counts']['train']} / {split_info['counts']['val']} / {split_info['counts']['test']}")
    logger.info(f"Speaker Breakdown:  {split_info['speakers']}")
    logger.info(f"Leakage Free:       {is_clean}")
    logger.info("=" * 60)

    return split_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Final Hybrid Telugu TTS Dataset")
    parser.add_argument("--indictts-man", type=str, default="dataset/telugu_tts/indictts_selected_manifest.json")
    parser.add_argument("--syspin-man", type=str, default="dataset/telugu_tts/syspin_selected_manifest.json")
    parser.add_argument("--output-dir", type=str, default="dataset/telugu_tts/processed_hybrid")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_hybrid_dataset(
        indictts_manifest=args.indictts_man,
        syspin_manifest=args.syspin_man,
        output_dir=args.output_dir,
        seed=args.seed
    )
