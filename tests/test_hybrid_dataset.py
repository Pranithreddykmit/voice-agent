"""
Unit tests for Hybrid Telugu Speech Dataset Assembly & Quality Control (Phase 6B.6).
Verifies multi-source integration (IndicTTS + SYSPIN), speaker tracking,
stratified splitting, leakage prevention, and IndicF5 dataloader compatibility.
"""

import os
import csv
import json
import tempfile
import shutil
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "dataset" / "telugu_tts" / "scripts"
import sys
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_hybrid_dataset import prepare_hybrid_dataset
from validate_dataset import dry_run_indicf5_dataloader, validate_wav_file


@pytest.fixture
def temp_hybrid_env():
    """Creates a temporary isolated environment with mock IndicTTS and SYSPIN data."""
    temp_dir = tempfile.mkdtemp(prefix="test_hybrid_tts_")
    base_path = Path(temp_dir)
    raw_indictts = base_path / "raw_indictts"
    raw_syspin = base_path / "raw_syspin"
    out_dir = base_path / "processed_hybrid"
    raw_indictts.mkdir(parents=True)
    raw_syspin.mkdir(parents=True)

    sr = 24000
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # 3 IndicTTS samples
    indictts_manifest = []
    for i in range(1, 4):
        sid = f"indictts_f_{i:04d}"
        wav_file = raw_indictts / f"{sid}.wav"
        sf.write(str(wav_file), signal, sr, subtype="PCM_16")
        indictts_manifest.append({
            "id": sid,
            "source": "indictts",
            "speaker": "indic_female",
            "category": "general",
            "text": f"ఇండిక్ టిటిఎస్ వాక్యం {i}.",
            "raw_wav": str(wav_file),
            "duration_sec": 2.0
        })

    # 4 SYSPIN samples (2 female, 2 male)
    syspin_manifest = []
    for i in range(1, 3):
        sid = f"syspin_f_{i:04d}"
        wav_file = raw_syspin / f"{sid}.wav"
        sf.write(str(wav_file), signal, sr, subtype="PCM_16")
        syspin_manifest.append({
            "id": sid,
            "source": "syspin",
            "speaker": "syspin_female",
            "category": "education",
            "text": f"సిస్పిన్ మహిళా విద్యా వాక్యం {i}.",
            "raw_wav": str(wav_file),
            "duration_sec": 2.0
        })

    for i in range(1, 3):
        sid = f"syspin_m_{i:04d}"
        wav_file = raw_syspin / f"{sid}.wav"
        sf.write(str(wav_file), signal, sr, subtype="PCM_16")
        syspin_manifest.append({
            "id": sid,
            "source": "syspin",
            "speaker": "syspin_male",
            "category": "science",
            "text": f"సిస్పిన్ పురుష శాస్త్ర వాక్యం {i}.",
            "raw_wav": str(wav_file),
            "duration_sec": 2.0
        })

    indictts_man_file = base_path / "indictts_manifest.json"
    syspin_man_file = base_path / "syspin_manifest.json"

    with open(indictts_man_file, "w", encoding="utf-8") as f:
        json.dump(indictts_manifest, f, ensure_ascii=False)
    with open(syspin_man_file, "w", encoding="utf-8") as f:
        json.dump(syspin_manifest, f, ensure_ascii=False)

    yield {
        "base_path": base_path,
        "indictts_man": str(indictts_man_file),
        "syspin_man": str(syspin_man_file),
        "out_dir": str(out_dir)
    }

    shutil.rmtree(temp_dir)


def test_prepare_hybrid_dataset_execution(temp_hybrid_env):
    split_info = prepare_hybrid_dataset(
        indictts_manifest=temp_hybrid_env["indictts_man"],
        syspin_manifest=temp_hybrid_env["syspin_man"],
        output_dir=temp_hybrid_env["out_dir"],
        seed=42
    )

    assert split_info["leakage_free"] is True
    assert split_info["counts"]["total"] == 7
    assert split_info["speakers"]["indic_female"] == 3
    assert split_info["speakers"]["syspin_female"] == 2
    assert split_info["speakers"]["syspin_male"] == 2

    # Check files created
    out_dir = Path(temp_hybrid_env["out_dir"])
    meta_csv = out_dir / "metadata.csv"
    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    test_csv = out_dir / "test.csv"
    manifest_json = out_dir / "hybrid_dataset_manifest.json"

    assert meta_csv.exists()
    assert train_csv.exists()
    assert val_csv.exists()
    assert test_csv.exists()
    assert manifest_json.exists()

    # Verify metadata header is strictly audio_file|text
    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        assert header == ["audio_file", "text"]
        rows = list(reader)
        assert len(rows) == 7


def test_hybrid_speaker_source_tracking(temp_hybrid_env):
    prepare_hybrid_dataset(
        indictts_manifest=temp_hybrid_env["indictts_man"],
        syspin_manifest=temp_hybrid_env["syspin_man"],
        output_dir=temp_hybrid_env["out_dir"],
        seed=42
    )

    manifest_json = Path(temp_hybrid_env["out_dir"]) / "hybrid_dataset_manifest.json"
    with open(manifest_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) == 7
    sources = {r["source"] for r in records}
    speakers = {r["speaker"] for r in records}

    assert sources == {"indictts", "syspin"}
    assert speakers == {"indic_female", "syspin_female", "syspin_male"}


def test_hybrid_indicf5_dataloader_compatibility(temp_hybrid_env):
    prepare_hybrid_dataset(
        indictts_manifest=temp_hybrid_env["indictts_man"],
        syspin_manifest=temp_hybrid_env["syspin_man"],
        output_dir=temp_hybrid_env["out_dir"],
        seed=42
    )

    meta_csv = Path(temp_hybrid_env["out_dir"]) / "metadata.csv"
    success, err_msg = dry_run_indicf5_dataloader(str(meta_csv))
    assert success is True
    assert err_msg is None
