"""
Audio Decoding, Normalization, and Serialization Utilities for Phase 5 Voice Subsystem.
Converts arbitrary browser audio (WAV, WebM, OGG, FLAC) to 16kHz mono PCM arrays.
"""

import io
import logging
import time
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import scipy.signal

logger = logging.getLogger(__name__)

# Minimum root-mean-square amplitude to distinguish speech from absolute silence
SILENCE_RMS_THRESHOLD = 0.002


class AudioProcessingError(Exception):
    """Raised when audio decoding, downmixing, or resampling fails."""
    pass


def decode_and_normalize_audio(
    audio_bytes: bytes,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int, float]:
    """
    Decodes raw audio bytes, downmixes to mono, and resamples to target_sr (default 16kHz).

    Args:
        audio_bytes: Raw audio byte buffer from HTTP multipart upload.
        target_sr: Target sample rate (16000 Hz for standard ASR models).

    Returns:
        (audio_array: np.ndarray (float32, 1D mono), sample_rate: int, duration_sec: float)
    """
    t0 = time.perf_counter()

    if not audio_bytes or len(audio_bytes) < 100:
        raise AudioProcessingError("Audio payload is empty or corrupted (less than 100 bytes).")

    try:
        # 1. Attempt reading with soundfile (supports WAV, FLAC, OGG, WebM/Vorbis)
        buf = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(buf, dtype="float32")
    except Exception as e1:
        # Try fallback using pydub if soundfile fails on browser container format
        try:
            from pydub import AudioSegment
            buf.seek(0)
            seg = AudioSegment.from_file(buf)
            seg = seg.set_frame_rate(target_sr).set_channels(1)
            raw_samples = seg.get_array_of_samples()
            data = np.array(raw_samples, dtype="float32") / (1 << (8 * seg.sample_width - 1))
            sample_rate = target_sr
        except Exception as e2:
            raise AudioProcessingError(
                f"Failed to decode audio file format. Primary error: {e1}; Fallback error: {e2}"
            )

    # 2. Downmix to mono if multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 3. Resample to target_sr if required (using polyphase filter with anti-aliasing)
    if sample_rate != target_sr:
        import math
        gcd = math.gcd(int(sample_rate), int(target_sr))
        up = target_sr // gcd
        down = sample_rate // gcd
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        sample_rate = target_sr

    duration_sec = len(data) / float(sample_rate)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        f"[Audio Processing] Decoded {len(audio_bytes)} bytes into {len(data)} mono samples "
        f"({duration_sec:.2f}s @ {sample_rate}Hz) in {elapsed_ms:.2f}ms."
    )

    return data, sample_rate, duration_sec


def is_silence(audio_array: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """Checks whether audio array is virtually silent or background noise."""
    if len(audio_array) == 0:
        return True
    rms = np.sqrt(np.mean(audio_array ** 2))
    return bool(rms < threshold)


def pcm_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 16000) -> bytes:
    """
    Serializes a 1D float32 or int16 numpy array to standard PCM 16-bit WAV bytes.
    """
    # Ensure float32 normalized in [-1.0, 1.0]
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    max_val = np.max(np.abs(audio_array))
    if max_val > 1.0:
        audio_array = audio_array / max_val

    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def prepare_speech_text(text: str, max_chars: int = 350) -> str:
    """
    Cleans markdown formatting, citations, and bullet points from answer text before speech synthesis.
    Ensures natural voice delivery and minimizes neural diffusion chunking latency.
    """
    import re
    if not text or not text.strip():
        return ""

    # Remove markdown citations like (Source: ..., Page X) or **(Source: ...)**
    clean = re.sub(r"\([*]*Source:[^)]*\)", "", text, flags=re.IGNORECASE)
    clean = re.sub(r"\*\*Source:[^*]*\*\*", "", clean, flags=re.IGNORECASE)

    # Strip markdown headers, bold, italics, code backticks, and bullet points
    clean = re.sub(r"[*#`_~]", "", clean)
    clean = re.sub(r"^\s*[-•*]\s+", "", clean, flags=re.MULTILINE)

    # Replace commas and semicolons with spaces to prevent excessive F5-TTS chunking
    clean = re.sub(r"[,;]\s*", " ", clean)

    # Collapse multiple whitespaces and newlines
    clean = re.sub(r"\s+", " ", clean).strip()

    # Truncate at sentence boundary if exceeding max_chars
    if len(clean) > max_chars:
        truncated = clean[:max_chars]
        last_stop = max(truncated.rfind("."), truncated.rfind("?"), truncated.rfind("!"), truncated.rfind("।"))
        if last_stop > 80:
            clean = truncated[:last_stop + 1]
        else:
            clean = truncated.strip() + "..."

    return clean

