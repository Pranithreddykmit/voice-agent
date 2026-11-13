"""
Unit tests for Telugu TTS Dataset Preparation Pipeline (Phase 6B).
Verifies audio preprocessing, duration filtering, metadata formatting,
stratified splitting, leakage prevention, and IndicF5 dataloader compatibility.
"""

import os
import io
import csv
import json
import tempfile
import shutil
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

# Add scripts directory to path for testing
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "dataset" / "telugu_tts" / "scripts"
import sys
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess import process_single_audio, resample_audio, trim_silence, run_preprocessing
from split_dataset import stratified_deterministic_split, verify_split_leakage, run_split
from validate_dataset import validate_wav_file, validate_csv_metadata, dry_run_indicf5_dataloader


@pytest.fixture
def temp_dataset_env():
    """Creates a temporary isolated dataset directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_telugu_tts_")
    base_path = Path(temp_dir)
    raw_path = base_path / "raw"
    proc_path = base_path / "processed"
    raw_path.mkdir(parents=True)
    proc_path.mkdir(parents=True)

    # Create dummy raw audio (48kHz stereo)
    sr_raw = 48000
    t = np.linspace(0, 2.0, int(2.0 * sr_raw), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    stereo_signal = np.stack([signal, signal], axis=-1)

    sample_ids = ["sample_0001", "sample_0002", "sample_0003", "sample_0004", "sample_0005"]
    for sid in sample_ids:
        sf.write(str(raw_path / f"{sid}.wav"), stereo_signal, sr_raw)

    # Create dummy sentence manifest
    manifest = [
        {"id": "sample_0001", "category": "general", "text": "నమస్కారం! బాగున్నారా?"},
        {"id": "sample_0002", "category": "academic", "text": "కనీసం 75% హాజరు తప్పనిసరి."},
        {"id": "sample_0003", "category": "technical_codemixed", "text": "ఈ RAG సిస్టమ్ వెక్టర్ డేటాబేస్ వాడుతుంది."},
        {"id": "sample_0004", "category": "numeric_percentage", "text": "2026 నాటికి 160 క్రెడిట్స్ పూర్తి చేయాలి."},
        {"id": "sample_0005", "category": "general", "text": "గ్రంథాలయం సాయంత్రం వరకు తెరిచి ఉంటుంది."}
    ]
    manifest_path = base_path / "sentence_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    yield base_path

    shutil.rmtree(temp_dir)


def test_audio_resampling_and_mono_downmixing(temp_dataset_env):
    raw_file = temp_dataset_env / "raw" / "sample_0001.wav"
    out_file = temp_dataset_env / "processed" / "wavs" / "sample_0001.wav"

    success, duration, err = process_single_audio(
        raw_path=str(raw_file),
        target_path=str(out_file),
        target_sr=24000
    )
    assert success is True
    assert err is None
    assert 1.8 <= duration <= 2.2

    # Verify output audio properties
    info = sf.info(str(out_file))
    assert info.samplerate == 24000
    assert info.channels == 1
    assert info.subtype == "PCM_16"


def test_duration_bounds_rejection(temp_dataset_env):
    too_short_path = temp_dataset_env / "raw" / "short.wav"
    too_long_path = temp_dataset_env / "raw" / "long.wav"
    out_path = temp_dataset_env / "processed" / "wavs" / "test.wav"

    sr = 24000
    # 0.1s audio (< 0.3s)
    short_audio = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(0.1 * sr)))).astype(np.float32)
    sf.write(str(too_short_path), short_audio, sr)

    # 35s audio (> 30s)
    long_audio = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 35.0, int(35.0 * sr)))).astype(np.float32)
    sf.write(str(too_long_path), long_audio, sr)

    # Test short rejection
    succ_s, dur_s, err_s = process_single_audio(str(too_short_path), str(out_path), target_sr=24000)
    assert succ_s is False
    assert "minimum" in err_s

    # Test long rejection
    succ_l, dur_l, err_l = process_single_audio(str(too_long_path), str(out_path), target_sr=24000)
    assert succ_l is False
    assert "maximum" in err_l


def test_preprocessing_pipeline_and_metadata_format(temp_dataset_env):
    summary = run_preprocessing(dataset_dir=str(temp_dataset_env))
    assert summary["successfully_processed"] == 5
    assert summary["skipped_files"] == 0
    assert summary["total_duration_sec"] > 0

    # Verify metadata.csv format (pipe-delimited, header audio_file|text)
    meta_csv = temp_dataset_env / "processed" / "metadata.csv"
    assert meta_csv.exists()

    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        assert header == ["audio_file", "text"]
        rows = list(reader)
        assert len(rows) == 5
        for r in rows:
            assert os.path.exists(r[0])
            assert len(r[1]) > 0


def test_stratified_deterministic_split_and_zero_leakage(temp_dataset_env):
    # Preprocess first
    run_preprocessing(dataset_dir=str(temp_dataset_env))

    # Run split with seed 42
    split_info = run_split(dataset_dir=str(temp_dataset_env), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    assert split_info["leakage_free"] is True
    assert split_info["counts"]["total"] == 5

    train_csv = temp_dataset_env / "processed" / "train.csv"
    val_csv = temp_dataset_env / "processed" / "val.csv"
    test_csv = temp_dataset_env / "processed" / "test.csv"
    assert train_csv.exists()
    assert val_csv.exists()
    assert test_csv.exists()


def test_leakage_detection_catches_duplicate_transcript():
    train_rows = [{"audio_file": "/wavs/1.wav", "text": "నమస్కారం! బాగున్నారా?"}]
    val_rows = [{"audio_file": "/wavs/2.wav", "text": "నమస్కారం! బాగున్నారా?"}]  # Duplicate transcript
    test_rows = [{"audio_file": "/wavs/3.wav", "text": "కనీసం 75% హాజరు తప్పనిసరి."}]

    is_clean, violations = verify_split_leakage(train_rows, val_rows, test_rows)
    assert is_clean is False
    assert len(violations) > 0
    assert "Transcript leakage Train-Val" in violations[0]


def test_indicf5_dataloader_dry_run_compatibility(temp_dataset_env):
    run_preprocessing(dataset_dir=str(temp_dataset_env))
    meta_csv = temp_dataset_env / "processed" / "metadata.csv"

    success, err_msg = dry_run_indicf5_dataloader(str(meta_csv))
    assert success is True
    assert err_msg is None
