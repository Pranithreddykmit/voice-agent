"""
Unit tests for Text-to-Speech (TTS) Providers (Phase 5 & 7).
Fast execution with full engine transparency and adapter validation.
"""

import pytest
from app.speech.tts import (
    FallbackTTSProvider,
    IndicF5TTSProvider,
    TTSSynthesisError,
    get_tts_provider
)


def test_fallback_tts_synthesis():
    """Verify fallback acoustic synthesis generates compliant WAV audio."""
    provider = FallbackTTSProvider(sample_rate=16000)
    text = "విశ్వవిద్యాలయ నిబంధనల ప్రకారం కనీసం 75% హాజరు తప్పనిసరి."

    wav_bytes, latency_ms = provider.synthesize(text=text, voice_id="telugu_female_1", language="te")
    assert len(wav_bytes) > 500
    assert wav_bytes.startswith(b"RIFF")
    assert latency_ms >= 0.0


def test_indicf5_tts_properties_and_fallback():
    """Verify IndicF5 provider exposes correct engine, adapter status, and fallback handling."""
    provider = IndicF5TTSProvider.__new__(IndicF5TTSProvider)
    provider.fallback = FallbackTTSProvider(sample_rate=16000)
    provider._model = None
    provider._blocker_reason = "Test offline simulation"
    provider._adapter_name = "none"
    provider._adapter_active = False

    assert "fallback_acoustic" in provider.engine_name
    assert provider.is_fallback is True
    assert provider.adapter_name == "none"
    assert provider.is_adapter_active is False

    wav_bytes, lat = provider.fallback.synthesize("నమస్కారం")
    assert len(wav_bytes) > 500
    assert wav_bytes.startswith(b"RIFF")


def test_tts_empty_text_error():
    """Verify empty text raises TTSSynthesisError."""
    provider = FallbackTTSProvider()
    with pytest.raises(TTSSynthesisError):
        provider.synthesize(text="")


def test_tts_factory():
    """Verify factory returns valid BaseTTSProvider."""
    tts = get_tts_provider()
    assert tts is not None
    assert isinstance(tts.engine_name, str)
    assert isinstance(tts.is_fallback, bool)
    assert hasattr(tts, "adapter_name")
    assert hasattr(tts, "is_adapter_active")


def test_tts_engine_transparency():
    """Verify engine name and fallback status transparency."""
    fallback_p = FallbackTTSProvider()
    assert fallback_p.engine_name == "fallback_acoustic"
    assert fallback_p.is_fallback is True
    assert fallback_p.adapter_name == "none"
    assert fallback_p.is_adapter_active is False
