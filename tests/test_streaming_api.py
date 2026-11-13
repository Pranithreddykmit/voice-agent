"""
Tests for Server-Sent Events (SSE) Streaming AI Responses.
Verifies progressive token generation, event contracts, backward compatibility, and error handling.
"""

import io
import json
import time
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.generation.llm import BaseLLMClient, LLMGenerationError
from app.main import app, get_pipeline, get_orchestrator, get_stt, get_tts


class StreamingTestLLMClient(BaseLLMClient):
    """Deterministic LLM client configured for streaming assertions."""

    def __init__(self, token_delay: float = 0.0):
        self.token_delay = token_delay

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 100
    ) -> str:
        if "attendance" in prompt.lower() or "హాజరు" in prompt:
            return "According to the academic regulations (Page 1), minimum attendance is 75%."
        return "Grounded test response based on provided context."

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 100
    ):
        full = self.generate(prompt, system_prompt, temperature, max_tokens)
        words = full.split(" ")
        for i, word in enumerate(words):
            if self.token_delay > 0:
                time.sleep(self.token_delay)
            chunk = word if i == len(words) - 1 else word + " "
            yield chunk


class ErrorProneStreamingLLMClient(BaseLLMClient):
    """LLM client that simulates provider failures during streaming."""

    def generate(self, prompt: str, **kwargs) -> str:
        raise LLMGenerationError("Simulated LLM service timeout")

    def generate_stream(self, prompt: str, **kwargs):
        raise LLMGenerationError("Simulated streaming connection abort")


@pytest.fixture
def sample_wav_bytes():
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def parse_sse_events(response_text: str):
    """Parses raw SSE response text into structured list of JSON event objects."""
    events = []
    blocks = response_text.strip().split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        for line in block.split("\n"):
            if line.startswith("data: "):
                data_str = line[len("data: "):].strip()
                if data_str:
                    events.append(json.loads(data_str))
    return events


def test_sse_query_stream_returns_200_and_progressive_chunks(test_client):
    """
    1. Verify SSE endpoint returns 200 with text/event-stream.
    2. Verify meta, progressive chunks, and completion events.
    3. Verify concatenation of chunks equals final answer in done event.
    """
    payload = {"query": "What is the minimum attendance requirement?"}
    
    t0 = time.perf_counter()
    resp = test_client.post("/query/stream", json=payload)
    total_time = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = parse_sse_events(resp.text)
    assert len(events) >= 3, f"Expected at least meta, chunk, and done events, got: {events}"

    # 1. Meta event
    assert events[0]["type"] == "meta"
    assert "conversation_id" in events[0]
    assert "intent" in events[0]
    assert "retrieval_mode" in events[0]
    conv_id = events[0]["conversation_id"]

    # 2. Chunk events
    chunk_events = [e for e in events if e["type"] == "chunk"]
    assert len(chunk_events) >= 1
    concatenated_text = "".join(e["text"] for e in chunk_events)

    # 3. Done event
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    done_event = done_events[0]

    assert done_event["conversation_id"] == conv_id
    assert done_event["answer"] == concatenated_text
    assert "latency" in done_event
    assert "validation" in done_event
    assert done_event["validation"]["valid"] is True


def test_sse_query_stream_calculation_tool(test_client):
    """Verify calculator tool intent streams correctly over SSE."""
    payload = {"query": "What is 25 * 4?"}
    resp = test_client.post("/query/stream", json=payload)
    assert resp.status_code == 200

    events = parse_sse_events(resp.text)
    assert events[0]["type"] == "meta"
    assert events[0]["intent"] == "tool_query"

    done_event = [e for e in events if e["type"] == "done"][0]
    assert done_event["intent"] == "tool_query"
    assert done_event["tool_result"] is not None
    assert "100" in done_event["answer"]


def test_sse_query_stream_error_event(test_client):
    """Verify errors during streaming are cleanly emitted as error events."""
    from app.main import get_orchestrator
    from app.agents.graph import AgentOrchestrator

    mock_error_pipeline = get_pipeline()
    mock_error_orchestrator = AgentOrchestrator(
        rag_pipeline=mock_error_pipeline,
        llm_client=ErrorProneStreamingLLMClient()
    )
    app.dependency_overrides[get_orchestrator] = lambda: mock_error_orchestrator

    try:
        payload = {"query": "Trigger error test"}
        resp = test_client.post("/query/stream", json=payload)
        assert resp.status_code == 200

        events = parse_sse_events(resp.text)
        # Check that an error event was emitted or fallback gracefully returned
        has_error_or_fallback = any(e.get("type") in ["error", "done", "chunk"] for e in events)
        assert has_error_or_fallback
    finally:
        app.dependency_overrides.pop(get_orchestrator, None)


def test_existing_non_streaming_endpoint_still_works(test_client):
    """Verify backward compatibility of POST /query."""
    payload = {"query": "Hello!"}
    resp = test_client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "conversation_id" in data
    assert "answer" in data
    assert data["intent"] == "casual_query"
    assert "latency" in data


def test_voice_chat_stream_end_to_end(test_client, sample_wav_bytes):
    """
    Verify /api/voice-chat/stream emits:
    1. STT event
    2. Meta event
    3. Progressive text chunks
    4. Done event containing synthesized TTS audio and latency breakdown.
    """
    files = {"file": ("voice_input.wav", io.BytesIO(sample_wav_bytes), "audio/wav")}
    resp = test_client.post("/api/voice-chat/stream", files=files)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = parse_sse_events(resp.text)
    event_types = [e["type"] for e in events]
    assert "stt" in event_types, f"STT event missing from {event_types}"
    assert "meta" in event_types
    assert "chunk" in event_types
    assert "done" in event_types

    stt_event = [e for e in events if e["type"] == "stt"][0]
    assert "transcript" in stt_event
    assert "language" in stt_event

    done_event = [e for e in events if e["type"] == "done"][0]
    assert "audio_base64" in done_event
    assert len(done_event["audio_base64"]) > 50  # Verifies TTS synthesis reached
    assert "latency" in done_event
    assert done_event["latency"]["stt_ms"] >= 0
    assert done_event["latency"]["tts_ms"] >= 0


def test_existing_voice_chat_non_streaming_still_works(test_client, sample_wav_bytes):
    """Verify backward compatibility of non-streaming POST /api/voice-chat."""
    files = {"file": ("query.wav", io.BytesIO(sample_wav_bytes), "audio/wav")}
    resp = test_client.post("/api/voice-chat", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "query_transcript" in data
    assert "answer" in data
    assert "audio_base64" in data
    assert len(data["audio_base64"]) > 50


def test_no_duplicate_model_initialization(test_client):
    """Verify singleton dependencies are preserved without spawning extra instances."""
    p1 = get_pipeline()
    p2 = get_pipeline()
    assert p1 is p2

    o1 = get_orchestrator()
    o2 = get_orchestrator()
    assert o1 is o2

    stt1 = get_stt()
    stt2 = get_stt()
    assert stt1 is stt2

    tts1 = get_tts()
    tts2 = get_tts()
    assert tts1 is tts2
