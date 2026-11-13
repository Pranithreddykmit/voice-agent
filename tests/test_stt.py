"""
Unit tests for Speech-to-Text (STT) Providers (Phase 5).
"""

import pytest
import numpy as np
import torch
from app.speech.stt import MockSTTProvider, WhisperSTTProvider, get_stt_provider


def test_mock_stt_transcription():
    provider = MockSTTProvider()

    # Active audio signal
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    transcript, lang, latency = provider.transcribe(audio, sample_rate=16000)
    assert len(transcript) > 0
    assert lang == "te"
    assert latency >= 0.0


def test_mock_stt_silence_handling():
    provider = MockSTTProvider()
    silent_audio = np.zeros(16000, dtype=np.float32)

    transcript, _, latency = provider.transcribe(silent_audio, sample_rate=16000)
    assert transcript == ""
    assert latency >= 0.0


def test_stt_factory():
    stt = get_stt_provider()
    assert stt is not None


def test_whisper_sanitize_transcript():
    # 1. Reject repeating syllable loops
    assert WhisperSTTProvider.sanitize_transcript("కాకాకాకాకాకాకాకాకాకా") == ""
    assert WhisperSTTProvider.sanitize_transcript("దాదాదాదాదాదాదాదా") == ""
    # 2. Collapse repetitive words
    assert WhisperSTTProvider.sanitize_transcript("hello hello hello world") == "hello world"
    # 3. Preserve authentic multi-word queries
    clean_telugu = "అటెండెన్స్ నియమాలు ఏమిటి?"
    assert WhisperSTTProvider.sanitize_transcript(clean_telugu) == clean_telugu
    clean_english = "What is the detention rule?"
    assert WhisperSTTProvider.sanitize_transcript(clean_english) == clean_english


def test_whisper_probe_target_language_mocked():
    provider = WhisperSTTProvider(model_name="mock")
    # Verify mock fallback behavior
    audio = np.random.randn(16000).astype(np.float32)
    txt, lang, lat = provider.transcribe(audio, sample_rate=16000)
    assert lat >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Heavy 800M Whisper CPU execution reserved for evaluation/evaluate_speech.py")
def test_whisper_stt_provider_cuda():
    provider = WhisperSTTProvider()
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    transcript, lang, latency = provider.transcribe(audio, sample_rate=16000, language="te")
    assert lang == "te"
    assert latency >= 0.0


