"""
Tests for LLM Client Factory and Gemini Provider.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.config import settings
from app.generation.llm import (
    GeminiLLMClient,
    MockLLMClient,
    get_llm_client,
    LLMGenerationError
)


def test_gemini_client_missing_key_raises(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    client = GeminiLLMClient(model_name="gemini-1.5-flash", api_key=None)
    with pytest.raises(LLMGenerationError) as exc_info:
        client.generate("Hello")
    assert "GEMINI_API_KEY is not configured" in str(exc_info.value)



def test_gemini_client_successful_generation():
    client = GeminiLLMClient(model_name="gemini-1.5-flash", api_key="dummy-test-key")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "నమస్కారం! నేను మీకు ఎలా సహాయపడగలను?"}]
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = client.generate("నమస్కారం", system_prompt="You are a helpful assistant.")
        assert result == "నమస్కారం! నేను మీకు ఎలా సహాయపడగలను?"
        assert mock_post.called
        call_args = mock_post.call_args
        assert "generativelanguage.googleapis.com" in call_args[0][0]
        assert call_args[1]["headers"]["x-goog-api-key"] == "dummy-test-key"



def test_get_llm_client_gemini_fallback_when_no_key(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    client = get_llm_client()
    assert isinstance(client, MockLLMClient)


def test_get_llm_client_gemini_with_key(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "dummy-key-12345")
    monkeypatch.setattr(settings, "LLM_MODEL", "gemini-1.5-flash")
    client = get_llm_client()
    assert isinstance(client, GeminiLLMClient)
    assert client.model_name == "gemini-1.5-flash"
    assert client.api_key == "dummy-key-12345"
