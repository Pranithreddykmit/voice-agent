"""
Focused Unit Tests for TTS LoRA Adapter Integration and API Metadata (Phase 7).
Tests:
- Configuration validation (TTS_ADAPTER = 'none' vs valid path)
- Fallback provider metadata and status transparency
- Adapter path resolution
- API response headers (X-TTS-Adapter, X-TTS-Engine, X-TTS-Is-Fallback)
- /health endpoint adapter metadata exposure
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.config import settings
from app.speech.tts import (
    FallbackTTSProvider,
    IndicF5TTSProvider,
    BaseTTSProvider,
    get_tts_provider
)
from app.schemas import TTSRequest, HealthResponse
from app.main import text_to_speech, health_check


def test_fallback_tts_adapter_properties():
    """Verify fallback provider reports 'none' for adapter and is_fallback=True."""
    provider = FallbackTTSProvider(sample_rate=16000)
    assert provider.engine_name == "fallback_acoustic"
    assert provider.is_fallback is True
    assert provider.adapter_name == "none"
    assert provider.is_adapter_active is False

    wav_bytes, lat_ms = provider.synthesize("నమస్కారం")
    assert len(wav_bytes) > 500
    assert wav_bytes.startswith(b"RIFF")
    assert lat_ms >= 0.0


def test_adapter_path_resolution():
    """Verify IndicF5 adapter path resolution handles workspace-relative and 'none' paths."""
    provider = IndicF5TTSProvider.__new__(IndicF5TTSProvider)
    # Test 'none'
    assert provider._resolve_adapter_path("none") is None
    assert provider._resolve_adapter_path("") is None

    # Test existing local adapter path
    adapter_path = "models/indicf5_telugu_lora"
    resolved = provider._resolve_adapter_path(adapter_path)
    assert resolved is not None
    assert Path(resolved).exists()
    assert Path(resolved).name == "indicf5_telugu_lora"


@pytest.mark.asyncio
async def test_text_to_speech_endpoint_headers():
    """Verify text_to_speech endpoint returns X-TTS-Adapter header and correct content."""
    provider = FallbackTTSProvider(sample_rate=16000)
    request = TTSRequest(text="నమస్కారం!", voice_id="default", language="te")

    response = await text_to_speech(request=request, tts=provider)
    assert response.status_code == 200
    assert response.media_type == "audio/wav"
    assert response.headers["X-TTS-Engine"] == "fallback_acoustic"
    assert response.headers["X-TTS-Adapter"] == "none"
    assert response.headers["X-TTS-Is-Fallback"] == "True"
    assert "X-TTS-Latency-Ms" in response.headers
    assert len(response.body) > 500


@pytest.mark.asyncio
async def test_health_check_endpoint_adapter_metadata():
    """Verify health_check endpoint exposes tts_engine, tts_adapter, and tts_adapter_active."""
    mock_pipeline = MagicMock()
    mock_pipeline.vector_store.is_connected.return_value = True

    mock_db = MagicMock()
    mock_db.is_connected.return_value = True

    mock_tts = FallbackTTSProvider(sample_rate=16000)

    health_res = await health_check(pipeline=mock_pipeline, db=mock_db, tts=mock_tts)
    assert isinstance(health_res, HealthResponse)
    assert health_res.tts_engine == "fallback_acoustic"
    assert health_res.tts_adapter == "none"
    assert health_res.tts_adapter_active is False
