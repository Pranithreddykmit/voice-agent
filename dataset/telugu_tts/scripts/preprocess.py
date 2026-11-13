"""
Audio Preprocessing & Metadata Generation Pipeline for Telugu TTS Dataset (Phase 6B).
Normalizes raw multi-format recordings to 24kHz mono 16-bit PCM WAV files,
applies duration filtering (0.3s - 30.0s), and writes metadata.csv for IndicF5 training.
"""

import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import soundfile as sf
import scipy.signal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset.preprocess")

TARGET_SAMPLE_RATE = 24000
MIN_DURATION_SEC = 0.3
MAX_DURATION_SEC = 30.0
SILENCE_THRESHOLD_RMS = 0.002


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamples 1D float32 audio array to target sample rate using Fourier method."""
    if orig_sr == target_sr:
        return audio
    num_target_samples = int(round(len(audio) * float(target_sr) / float(orig_sr)))
    resampled = scipy.signal.resample(audio, num_target_samples)
    return resampled.astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = 25,
    threshold: float = SILENCE_THRESHOLD_RMS
) -> np.ndarray:
    """Trims excessive leading and trailing silence with smooth 20ms cosine fade."""
    frame_length = int(sr * (frame_ms / 1000.0))
    if len(audio) < frame_length * 2:
        return audio

    # Compute moving RMS energy
    num_frames = len(audio) // frame_length
    rms_values = np.zeros(num_frames)
    for i in range(num_frames):
        frame = audio[i * frame_length : (i + 1) * frame_length]
        rms_values[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

    active_frames = np.where(rms_values >= threshold)[0]
    if len(active_frames) == 0:
        return audio

    # Add 2 padding frames (50ms) buffer
    start_frame = max(0, active_frames[0] - 2)
    end_frame = min(num_frames, active_frames[-1] + 3)

    start_idx = start_frame * frame_length
    end_idx = min(len(audio), end_frame * frame_length)
    trimmed = audio[start_idx:end_idx]

    # Apply 20ms cosine fade-in and fade-out
    fade_len = min(int(sr * 0.02), len(trimmed) // 4)
    if fade_len > 0:
        fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
        fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))
        trimmed[:fade_len] *= fade_in
        trimmed[-fade_len:] *= fade_out

    return trimmed


def process_single_audio(
    raw_path: str,
    target_path: str,
    target_sr: int = TARGET_SAMPLE_RATE
) -> Tuple[bool, float, Optional[str]]:
    """
    Decodes, downmixes to mono, resamples to 24kHz, trims silence, and saves 16-bit PCM WAV.
    Returns:
        (success, duration_sec, error_message)
    """
    try:
        data, orig_sr = sf.read(raw_path, dtype="float32")

        # Downmix multi-channel to mono
        if data.ndim > 1:
            data = np.mean(data, axis=-1)

        # Normalize DC offset
        data = data - np.mean(data)

        # Check silence
        rms = np.sqrt(np.mean(data ** 2) + 1e-12)
        if rms < SILENCE_THRESHOLD_RMS:
            return False, 0.0, f"Audio is silent (RMS={rms:.6f} < {SILENCE_THRESHOLD_RMS})"

        # Resample to target sample rate (24kHz)
        if orig_sr != target_sr:
            data = resample_audio(data, orig_sr, target_sr)

        # Trim excessive silence
        data = trim_silence(data, target_sr)

        # Duration check
        duration_sec = len(data) / float(target_sr)
        if duration_sec < MIN_DURATION_SEC:
            return False, duration_sec, f"Duration {duration_sec:.2f}s < minimum {MIN_DURATION_SEC}s"
        if duration_sec > MAX_DURATION_SEC:
            return False, duration_sec, f"Duration {duration_sec:.2f}s > maximum {MAX_DURATION_SEC}s"

        # Peak normalization if near clipping
        peak = np.max(np.abs(data))
        if peak > 0.98:
            data = (data / peak) * 0.95

        # Save 16-bit PCM WAV
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        sf.write(target_path, data, target_sr, format="WAV", subtype="PCM_16")

        return True, round(duration_sec, 2), None

    except Exception as e:
        return False, 0.0, f"Audio processing error: {str(e)}"


def run_preprocessing(
    dataset_dir: str,
    raw_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    manifest_path: Optional[str] = None
) -> Dict[str, Any]:
    """Executes the dataset preprocessing pipeline."""
    base_dir = Path(dataset_dir)
    raw_path = Path(raw_dir) if raw_dir else base_dir / "raw"
    proc_path = Path(processed_dir) if processed_dir else base_dir / "processed"
    wavs_out_path = proc_path / "wavs"
    man_path = Path(manifest_path) if manifest_path else base_dir / "sentence_manifest.json"

    wavs_out_path.mkdir(parents=True, exist_ok=True)

    if not man_path.exists():
        raise FileNotFoundError(f"Sentence manifest not found at: {man_path}")

    with open(man_path, "r", encoding="utf-8") as f:
        manifest: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(manifest)} sentence templates from {man_path.name}")
    logger.info(f"Scanning raw audio files in: {raw_path}")

    processed_records = []
    metadata_rows = []
    skipped_count = 0
    total_processed_duration = 0.0

    raw_files = {p.stem: p for p in raw_path.glob("*.*") if p.suffix.lower() in [".wav", ".m4a", ".mp3", ".flac", ".ogg"]}

    for item in manifest:
        sample_id = item["id"]
        text = item["text"].strip()
        cat = item.get("category", "general")

        if sample_id not in raw_files:
            continue

        raw_file_path = raw_files[sample_id]
        target_wav_path = wavs_out_path / f"{sample_id}.wav"

        success, duration, error_msg = process_single_audio(
            raw_path=str(raw_file_path),
            target_path=str(target_wav_path),
            target_sr=TARGET_SAMPLE_RATE
        )

        if not success:
            logger.warning(f"Skipping {sample_id}: {error_msg}")
            skipped_count += 1
            continue

        total_processed_duration += duration
        metadata_rows.append({
            "audio_file": str(target_wav_path.resolve()),
            "text": text
        })

        processed_records.append({
            "id": sample_id,
            "category": cat,
            "text": text,
            "audio_file": str(target_wav_path.resolve()),
            "duration_sec": duration,
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": 1
        })

    # Write official IndicF5 training metadata.csv (pipe-delimited)
    metadata_csv_path = proc_path / "metadata.csv"
    with open(metadata_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for row in metadata_rows:
            writer.writerow([row["audio_file"], row["text"]])

    # Write rich dataset manifest
    dataset_manifest_path = proc_path / "dataset_manifest.json"
    with open(dataset_manifest_path, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=2, ensure_ascii=False)

    summary = {
        "manifest_sentences": len(manifest),
        "raw_files_found": len(raw_files),
        "successfully_processed": len(processed_records),
        "skipped_files": skipped_count,
        "total_duration_sec": round(total_processed_duration, 2),
        "total_duration_min": round(total_processed_duration / 60.0, 2),
        "metadata_csv": str(metadata_csv_path),
        "dataset_manifest": str(dataset_manifest_path)
    }

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETED")
    logger.info(f"Processed Utterances:   {summary['successfully_processed']} / {summary['manifest_sentences']}")
    logger.info(f"Total Audio Duration:   {summary['total_duration_sec']}s ({summary['total_duration_min']} min)")
    logger.info(f"Metadata CSV:           {summary['metadata_csv']}")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Telugu TTS Audio Dataset")
    parser.add_argument("--dataset-dir", type=str, default="dataset/telugu_tts", help="Base dataset directory")
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory containing raw audio files")
    parser.add_argument("--processed-dir", type=str, default=None, help="Directory to save processed audio")
    parser.add_argument("--manifest", type=str, default=None, help="Path to sentence manifest JSON")
    args = parser.parse_args()

    run_preprocessing(
        dataset_dir=args.dataset_dir,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        manifest_path=args.manifest
    )
