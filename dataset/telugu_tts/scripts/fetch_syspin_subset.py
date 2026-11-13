"""
Deterministic Subset Selector & Quality Filter for SYSPIN Telugu Dataset (Phase 6B.6).
Fetches audio and metadata via Hugging Face Dataset Server API, applies strict QC filtering,
and saves reproducible raw WAV files and manifest.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Any

import httpx
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset.fetch_syspin")

DATASET_NAME = "arpit-tiwari/syspin-telugu-tts"
HF_API_BASE = "https://datasets-server.huggingface.co/rows"

# QC Criteria
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 20.0
MIN_TEXT_CHARS = 10


def fetch_syspin_subset(
    output_dir: str = "dataset/telugu_tts/raw_syspin",
    manifest_out: str = "dataset/telugu_tts/syspin_selected_manifest.json",
    target_female: int = 40,
    target_male: int = 40,
    timeout_sec: float = 15.0
) -> Dict[str, Any]:
    """Fetches a high-quality, balanced subset of SYSPIN Telugu utterances with strict QC."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(timeout=timeout_sec)
    offset = 0
    batch_length = 100

    female_samples: List[Dict[str, Any]] = []
    male_samples: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()
    total_scanned = 0
    rejected_qc = 0

    logger.info(f"Connecting to Hugging Face Dataset Server for {DATASET_NAME}...")

    while (len(female_samples) < target_female or len(male_samples) < target_male) and offset < 2000:
        url = f"{HF_API_BASE}?dataset={DATASET_NAME}&config=default&split=train&offset={offset}&length={batch_length}"
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                logger.warning(f"API request failed with status {resp.status_code}, offset {offset}")
                break

            data = resp.json()
            rows = data.get("rows", [])
            if not rows:
                break

            for r in rows:
                total_scanned += 1
                row_data = r.get("row", {})
                gender = str(row_data.get("gender", "")).lower().strip()
                domain = str(row_data.get("domain", "GENERAL")).strip()
                sentence = str(row_data.get("sentence", "")).strip()
                audio_field = row_data.get("audio", [])

                # Get audio URL
                audio_url = None
                if isinstance(audio_field, list) and len(audio_field) > 0:
                    audio_url = audio_field[0].get("src")
                elif isinstance(audio_field, dict):
                    audio_url = audio_field.get("src")

                if not audio_url or not sentence or len(sentence) < MIN_TEXT_CHARS:
                    rejected_qc += 1
                    continue

                if sentence in seen_texts:
                    rejected_qc += 1
                    continue

                # Gender capacity check
                if gender == "female" and len(female_samples) >= target_female:
                    continue
                if gender == "male" and len(male_samples) >= target_male:
                    continue
                if gender not in ["female", "male"]:
                    continue

                # Download audio buffer directly from CDN URL
                try:
                    audio_resp = client.get(audio_url)
                    if audio_resp.status_code != 200:
                        rejected_qc += 1
                        continue

                    import io
                    audio_bytes = io.BytesIO(audio_resp.content)
                    audio_data, sr = sf.read(audio_bytes, dtype="float32")

                    # Duration & Audio QC Check
                    duration = len(audio_data) / float(sr)
                    if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
                        rejected_qc += 1
                        continue

                    peak = float(np.max(np.abs(audio_data)))
                    if peak > 0.99 or peak < 0.05:
                        rejected_qc += 1
                        continue

                    seen_texts.add(sentence)
                    item_info = {
                        "gender": gender,
                        "speaker": f"syspin_{gender}",
                        "domain": domain,
                        "sentence": sentence,
                        "audio_data": audio_data,
                        "orig_sr": sr,
                        "duration": round(duration, 2),
                        "peak": round(peak, 3)
                    }

                    if gender == "female":
                        female_samples.append(item_info)
                    else:
                        male_samples.append(item_info)

                    if len(female_samples) >= target_female and len(male_samples) >= target_male:
                        break

                except Exception as ex:
                    logger.debug(f"Audio fetch error: {ex}")
                    rejected_qc += 1
                    continue

            offset += batch_length

        except Exception as e:
            logger.error(f"Error querying Dataset Server: {e}")
            break

    client.close()

    # Save to disk and construct manifest
    all_selected = female_samples + male_samples
    manifest_records = []

    for idx, item in enumerate(all_selected, start=1):
        sample_id = f"syspin_{item['gender'][0]}_{idx:04d}"
        wav_file = out_path / f"{sample_id}.wav"
        sf.write(str(wav_file), item["audio_data"], item["orig_sr"], subtype="PCM_16")

        manifest_records.append({
            "id": sample_id,
            "source": "syspin",
            "speaker": item["speaker"],
            "category": item["domain"].lower().replace(" ", "_"),
            "domain_raw": item["domain"],
            "text": item["sentence"],
            "raw_wav": str(wav_file.resolve()),
            "duration_sec": item["duration"],
            "orig_sample_rate": item["orig_sr"]
        })

    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    total_duration = sum(r["duration_sec"] for r in manifest_records)
    summary = {
        "dataset_source": "arpit-tiwari/syspin-telugu-tts",
        "total_scanned": total_scanned,
        "rejected_qc": rejected_qc,
        "retained_samples": len(manifest_records),
        "female_count": len(female_samples),
        "male_count": len(male_samples),
        "total_duration_sec": round(total_duration, 2),
        "total_duration_min": round(total_duration / 60.0, 2),
        "manifest_path": manifest_out,
        "output_dir": str(out_path)
    }

    logger.info("=" * 60)
    logger.info("SYSPIN SUBSET ACQUISITION & QC COMPLETED")
    logger.info(f"Retained:       {summary['retained_samples']} samples ({summary['female_count']} female, {summary['male_count']} male)")
    logger.info(f"Total Duration: {summary['total_duration_sec']}s ({summary['total_duration_min']} min)")
    logger.info(f"QC Rejections:  {summary['rejected_qc']} / {summary['total_scanned']}")
    logger.info(f"Manifest:       {summary['manifest_path']}")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and QC SYSPIN Telugu Subset")
    parser.add_argument("--female", type=int, default=40, help="Female utterances count")
    parser.add_argument("--male", type=int, default=40, help="Male utterances count")
    parser.add_argument("--output-dir", type=str, default="dataset/telugu_tts/raw_syspin")
    parser.add_argument("--manifest-out", type=str, default="dataset/telugu_tts/syspin_selected_manifest.json")
    args = parser.parse_args()

    fetch_syspin_subset(
        output_dir=args.output_dir,
        manifest_out=args.manifest_out,
        target_female=args.female,
        target_male=args.male
    )
