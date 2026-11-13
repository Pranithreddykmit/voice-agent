"""
Deterministic Stratified Dataset Splitting & Leakage Verification (Phase 6B).
Splits processed dataset into Train (80%), Val (10%), and Test (10%) splits with fixed seed
and verifies zero leakage across splits based on normalized transcript identity.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset.split")


def normalize_transcript_for_leakage_check(text: str) -> str:
    """Normalizes transcript text to catch near-duplicate transcripts and whitespace variants."""
    import re
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[\s\.,!?;:\"'()\[\]{}]+", " ", cleaned)
    return cleaned.strip()


def verify_split_leakage(
    train_rows: List[Dict[str, str]],
    val_rows: List[Dict[str, str]],
    test_rows: List[Dict[str, str]]
) -> Tuple[bool, List[str]]:
    """
    Verifies that no exact or near-identical transcripts leak between splits.
    Returns:
        (is_leak_free, list_of_violations)
    """
    train_texts: Set[str] = {normalize_transcript_for_leakage_check(r["text"]) for r in train_rows}
    val_texts: Set[str] = {normalize_transcript_for_leakage_check(r["text"]) for r in val_rows}
    test_texts: Set[str] = {normalize_transcript_for_leakage_check(r["text"]) for r in test_rows}

    train_files: Set[str] = {r.get("audio_file") or r.get("audio_path", "") for r in train_rows}
    val_files: Set[str] = {r.get("audio_file") or r.get("audio_path", "") for r in val_rows}
    test_files: Set[str] = {r.get("audio_file") or r.get("audio_path", "") for r in test_rows}

    violations = []

    # File identity overlap
    f_train_val = train_files.intersection(val_files)
    f_train_test = train_files.intersection(test_files)
    f_val_test = val_files.intersection(test_files)
    if f_train_val:
        violations.append(f"Audio file overlap Train-Val: {f_train_val}")
    if f_train_test:
        violations.append(f"Audio file overlap Train-Test: {f_train_test}")
    if f_val_test:
        violations.append(f"Audio file overlap Val-Test: {f_val_test}")

    # Transcript identity overlap
    t_train_val = train_texts.intersection(val_texts)
    t_train_test = train_texts.intersection(test_texts)
    t_val_test = val_texts.intersection(test_texts)
    if t_train_val:
        violations.append(f"Transcript leakage Train-Val: {t_train_val}")
    if t_train_test:
        violations.append(f"Transcript leakage Train-Test: {t_train_test}")
    if t_val_test:
        violations.append(f"Transcript leakage Val-Test: {t_val_test}")

    is_leak_free = len(violations) == 0
    return is_leak_free, violations


def stratified_deterministic_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits records deterministically with stratification across categories."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-4, "Split ratios must sum to 1.0"

    rng = random.Random(seed)

    # Group by category
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        cat = r.get("category", "general")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    train_records = []
    val_records = []
    test_records = []

    # Deterministically shuffle within each category and partition
    for cat in sorted(categories.keys()):
        items = list(categories[cat])
        rng.shuffle(items)

        n_total = len(items)
        n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 5 else 0
        n_test = max(1, int(round(n_total * test_ratio))) if n_total >= 5 else 0
        n_train = n_total - n_val - n_test

        cat_train = items[:n_train]
        cat_val = items[n_train : n_train + n_val]
        cat_test = items[n_train + n_val :]

        train_records.extend(cat_train)
        val_records.extend(cat_val)
        test_records.extend(cat_test)

    return train_records, val_records, test_records


def write_split_csv(file_path: Path, records: List[Dict[str, Any]]):
    """Writes a split CSV in official IndicF5 pipe-delimited format (audio_file|text)."""
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for r in records:
            writer.writerow([r["audio_file"], r["text"]])


def run_split(
    dataset_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Any]:
    """Executes the dataset splitting and leakage validation workflow."""
    proc_path = Path(dataset_dir) / "processed"
    manifest_file = proc_path / "dataset_manifest.json"
    metadata_csv = proc_path / "metadata.csv"

    if not manifest_file.exists() and not metadata_csv.exists():
        raise FileNotFoundError(f"Processed dataset not found in {proc_path}. Run preprocess.py first.")

    # Load records
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            records: List[Dict[str, Any]] = json.load(f)
    else:
        records = []
        with open(metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                records.append({"audio_file": row["audio_file"], "text": row["text"], "category": "general"})

    if len(records) == 0:
        logger.warning("Dataset is empty. 0 records found to split.")
        return {"status": "empty", "total": 0}

    # Execute Stratified Split
    train_records, val_records, test_records = stratified_deterministic_split(
        records=records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # Leakage Verification
    is_clean, violations = verify_split_leakage(train_records, val_records, test_records)
    if not is_clean:
        logger.error(f"DATASET LEAKAGE DETECTED ({len(violations)} violations):")
        for v in violations:
            logger.error(f"  - {v}")
        raise ValueError("Dataset split failed due to data leakage!")

    # Write output split files
    train_csv = proc_path / "train.csv"
    val_csv = proc_path / "val.csv"
    test_csv = proc_path / "test.csv"
    split_info_json = proc_path / "split_info.json"

    write_split_csv(train_csv, train_records)
    write_split_csv(val_csv, val_records)
    write_split_csv(test_csv, test_records)

    split_info = {
        "random_seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "leakage_free": is_clean,
        "counts": {
            "total": len(records),
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        },
        "durations_sec": {
            "total": round(sum(r.get("duration_sec", 0) for r in records), 2),
            "train": round(sum(r.get("duration_sec", 0) for r in train_records), 2),
            "val": round(sum(r.get("duration_sec", 0) for r in val_records), 2),
            "test": round(sum(r.get("duration_sec", 0) for r in test_records), 2)
        }
    }

    with open(split_info_json, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    logger.info("=" * 60)
    logger.info("DATASET SPLIT COMPLETED (Zero Leakage Verified)")
    logger.info(f"Random Seed:    {seed}")
    logger.info(f"Train Records:  {len(train_records)} ({split_info['durations_sec']['train']}s)")
    logger.info(f"Val Records:    {len(val_records)} ({split_info['durations_sec']['val']}s)")
    logger.info(f"Test Records:   {len(test_records)} ({split_info['durations_sec']['test']}s)")
    logger.info(f"Train ∩ Val = ∅ | Train ∩ Test = ∅ | Val ∩ Test = ∅ : VERIFIED")
    logger.info("=" * 60)

    return split_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Telugu TTS Dataset into Train/Val/Test")
    parser.add_argument("--dataset-dir", type=str, default="dataset/telugu_tts", help="Base dataset directory")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_split(
        dataset_dir=args.dataset_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
