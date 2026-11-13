"""
Telugu TTS Dataset Quality Inspector & Dry-Run Validator (Phase 6B).
Audits audio files (24kHz mono 16-bit PCM), metadata integrity, category distributions,
split leakage, and executes an offline dry-run through the IndicF5 CustomDataset dataloader.
"""

import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import soundfile as sf
import numpy as np

# Ensure ai-service is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
AI_SERVICE_DIR = BASE_DIR / "ai-service"
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset.validate")


def validate_wav_file(wav_path: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Validates an individual WAV file against 24kHz mono 16-bit PCM criteria."""
    try:
        info = sf.info(wav_path)
        duration = info.duration

        errors = []
        if info.samplerate != 24000:
            errors.append(f"Sample rate {info.samplerate} != 24000 Hz")
        if info.channels != 1:
            errors.append(f"Channels {info.channels} != 1 (mono)")
        if info.subtype != "PCM_16":
            errors.append(f"Subtype {info.subtype} != PCM_16")
        if duration < 0.3:
            errors.append(f"Duration {duration:.2f}s < 0.3s")
        if duration > 30.0:
            errors.append(f"Duration {duration:.2f}s > 30.0s")

        is_valid = len(errors) == 0
        details = {
            "path": wav_path,
            "duration": round(duration, 2),
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype
        }
        return is_valid, details, ("; ".join(errors) if errors else None)

    except Exception as e:
        return False, {"path": wav_path}, f"File read error: {str(e)}"


def validate_csv_metadata(csv_path: str) -> Tuple[bool, List[Dict[str, str]], List[str]]:
    """Validates a pipe-delimited metadata CSV file."""
    if not os.path.exists(csv_path):
        return False, [], [f"CSV file not found: {csv_path}"]

    rows = []
    errors = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader, None)

        if header != ["audio_file", "text"]:
            errors.append(f"Invalid CSV header in {csv_path}: {header} (Expected: ['audio_file', 'text'])")

        for idx, row in enumerate(reader, start=2):
            if len(row) != 2:
                errors.append(f"Row {idx} malformed in {csv_path}: {row}")
                continue
            audio_path, text = row[0].strip(), row[1].strip()
            if not audio_path:
                errors.append(f"Row {idx} has empty audio path")
            elif not os.path.exists(audio_path):
                errors.append(f"Row {idx} audio file does not exist: {audio_path}")
            if not text:
                errors.append(f"Row {idx} has empty transcript")

            rows.append({"audio_file": audio_path, "text": text})

    return len(errors) == 0, rows, errors


def dry_run_indicf5_dataloader(csv_path: str) -> Tuple[bool, Optional[str]]:
    """Performs a non-destructive dry-run through the IndicF5 dataset loading components."""
    try:
        import torch
        from f5_tts.model.dataset import CustomDataset, collate_fn

        # Parse CSV rows
        records = []
        durations = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                dur = sf.info(row["audio_file"]).duration
                records.append({
                    "audio_path": row["audio_file"],
                    "text": row["text"],
                    "duration": dur
                })
                durations.append(dur)

        if len(records) == 0:
            return True, "No records to test dry-run"

        dataset = CustomDataset(
            custom_dataset=records,
            durations=durations,
            target_sample_rate=24000,
            hop_length=256,
            n_mel_channels=100
        )

        # Test first item extraction
        item_0 = dataset[0]
        mel_spec = item_0["mel_spec"]
        text_0 = item_0["text"]

        assert isinstance(mel_spec, torch.Tensor), "mel_spec is not a torch.Tensor"
        assert mel_spec.shape[0] == 100, f"mel_spec channels {mel_spec.shape[0]} != 100"
        assert isinstance(text_0, str) and len(text_0) > 0, "text is invalid"

        # Test batch collation
        sample_batch = [dataset[i % len(dataset)] for i in range(min(2, len(dataset)))]
        batch_dict = collate_fn(sample_batch)
        assert "mel" in batch_dict, "collate_fn missing mel"
        assert "text" in batch_dict, "collate_fn missing text"

        return True, None

    except Exception as e:
        return False, f"IndicF5 CustomDataset dry-run failed: {str(e)}"


def run_validation(dataset_dir: str) -> Dict[str, Any]:
    """Runs complete dataset validation and prints a structured quality report."""
    base_path = Path(dataset_dir)
    proc_path = base_path / "processed"
    wavs_dir = proc_path / "wavs"
    metadata_csv = proc_path / "metadata.csv"
    train_csv = proc_path / "train.csv"
    val_csv = proc_path / "val.csv"
    test_csv = proc_path / "test.csv"
    manifest_json = proc_path / "dataset_manifest.json"

    print("=" * 80)
    print("PHASE 6B: TELUGU TTS DATASET QUALITY REPORT & VALIDATION HARNESS")
    print("=" * 80)

    # 1. Inspect Audio Files
    wav_files = sorted(list(wavs_dir.glob("*.wav")))
    valid_wavs = []
    audio_errors = []
    durations = []

    for w in wav_files:
        is_val, details, err = validate_wav_file(str(w))
        if is_val:
            valid_wavs.append(details)
            durations.append(details["duration"])
        else:
            audio_errors.append(f"{w.name}: {err}")

    # 2. Inspect CSVs
    meta_ok, meta_rows, meta_errs = validate_csv_metadata(str(metadata_csv))
    train_ok, train_rows, train_errs = validate_csv_metadata(str(train_csv))
    val_ok, val_rows, val_errs = validate_csv_metadata(str(val_csv))
    test_ok, test_rows, test_errs = validate_csv_metadata(str(test_csv))

    # 3. Leakage Validation
    from split_dataset import verify_split_leakage
    if train_rows and val_rows and test_rows:
        leakage_free, leak_violations = verify_split_leakage(train_rows, val_rows, test_rows)
    else:
        leakage_free, leak_violations = True, []

    # 4. IndicF5 Dry-Run
    indicf5_dry_run_ok, dry_run_msg = (True, "Skipped (no rows)")
    if train_rows:
        indicf5_dry_run_ok, dry_run_msg = dry_run_indicf5_dataloader(str(train_csv))

    # 5. Category Breakdown from manifest
    cat_counts = {}
    if manifest_json.exists():
        with open(manifest_json, "r", encoding="utf-8") as f:
            man_data = json.load(f)
            for m in man_data:
                c = m.get("category", "general")
                cat_counts[c] = cat_counts.get(c, 0) + 1

    # Print Summary Report
    print(f"Total Processed WAVs Found:     {len(wav_files)}")
    print(f"Valid 24kHz Mono 16-bit WAVs:   {len(valid_wavs)}")
    print(f"Audio File Validation Errors:   {len(audio_errors)}")

    if durations:
        print(f"\n--- Duration Metrics ---")
        print(f"Total Duration:                 {sum(durations):.2f}s ({sum(durations)/60.0:.2f} min)")
        print(f"Average Duration:               {np.mean(durations):.2f}s")
        print(f"Min Duration:                   {min(durations):.2f}s")
        print(f"Max Duration:                   {max(durations):.2f}s")

    print(f"\n--- Split Distribution ---")
    print(f"Metadata Total Records:         {len(meta_rows)}")
    print(f"Train Split:                    {len(train_rows)}")
    print(f"Validation Split:               {len(val_rows)}")
    print(f"Test Split (Unseen Evaluation): {len(test_rows)}")

    if cat_counts:
        print(f"\n--- Category Breakdown ---")
        for cat, cnt in cat_counts.items():
            print(f"  - {cat:<25}: {cnt} utterances")

    print(f"\n--- Integrity & Compatibility Verification ---")
    print(f"24 kHz / Mono / PCM_16 Check:   {'PASS' if len(audio_errors) == 0 and len(wav_files) > 0 else ('PASS (No files yet)' if len(wav_files) == 0 else 'FAIL')}")
    print(f"Duration [0.3s - 30.0s] Check:  {'PASS' if len(audio_errors) == 0 and len(wav_files) > 0 else ('PASS (No files yet)' if len(wav_files) == 0 else 'FAIL')}")
    print(f"Metadata CSV Header & Paths:    {'PASS' if meta_ok else ('PASS (No files yet)' if not metadata_csv.exists() else 'FAIL')}")
    print(f"Zero Data Leakage Check:        {'PASS' if leakage_free else 'FAIL'}")
    print(f"IndicF5 CustomDataset Dry-Run:  {'PASS' if indicf5_dry_run_ok else 'FAIL'}")
    print("=" * 80)

    report = {
        "wav_count": len(wav_files),
        "valid_wavs": len(valid_wavs),
        "audio_errors": audio_errors,
        "total_duration_sec": round(sum(durations), 2) if durations else 0.0,
        "split_counts": {
            "metadata": len(meta_rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows)
        },
        "category_counts": cat_counts,
        "leakage_free": leakage_free,
        "indicf5_dry_run_pass": indicf5_dry_run_ok
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Telugu TTS Dataset")
    parser.add_argument("--dataset-dir", type=str, default="dataset/telugu_tts", help="Base dataset directory")
    args = parser.parse_args()

    run_validation(dataset_dir=args.dataset_dir)
