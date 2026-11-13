"""
Unit tests for Audio Processing and Normalization Utilities (Phase 5).
"""

import io
import pytest
import numpy as np
import soundfile as sf

from app.speech.audio import (
    decode_and_normalize_audio,
    is_silence,
    pcm_to_wav_bytes,
    AudioProcessingError
)


def test_audio_decode_and_resample():
    # Create 1 second of 44.1kHz stereo audio
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    stereo = np.vstack([left, right]).T

    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    raw_bytes = buf.getvalue()

    # Decode and resample to 16kHz mono
    data, target_sr, duration = decode_and_normalize_audio(raw_bytes, target_sr=16000)
    assert target_sr == 16000
    assert data.ndim == 1  # mono
    assert len(data) == 16000
    assert abs(duration - 1.0) < 0.05


def test_silence_detection():
    # Silent audio array (zeros)
    silent_audio = np.zeros(16000, dtype=np.float32)
    assert is_silence(silent_audio) is True

    # Active audio signal
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    active_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert is_silence(active_audio) is False


def test_pcm_to_wav_bytes():
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_bytes = pcm_to_wav_bytes(audio, sample_rate=16000)

    assert len(wav_bytes) > 100
    assert wav_bytes.startswith(b"RIFF")


def test_corrupted_audio_rejection():
    corrupted_bytes = b"NOT_A_VALID_AUDIO_HEADER_OR_CONTENT"
    with pytest.raises(AudioProcessingError):
        decode_and_normalize_audio(corrupted_bytes)


def test_prepare_speech_text_strips_markdown_and_citations():
    from app.speech.audio import prepare_speech_text
    markdown_text = (
        "Based on regulations, the **Detention Rule** is:\n"
        "* Students with < **65% attendance** are detained.\n"
        "(Source: academic_regulations.pdf, Page 1)"
    )
    clean = prepare_speech_text(markdown_text)
    assert "**" not in clean
    assert "(Source:" not in clean
    assert "Detention Rule" in clean
    assert "65% attendance" in clean

