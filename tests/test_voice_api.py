"""
API endpoint tests for Phase 5 Voice Subsystem (STT, TTS, and Composite Voice-Chat).
"""

import io
import pytest
import numpy as np
import soundfile as sf


@pytest.fixture
def sample_wav_bytes():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def test_api_stt_endpoint(test_client, sample_wav_bytes):
    files = {"file": ("recording.wav", io.BytesIO(sample_wav_bytes), "audio/wav")}
    resp = test_client.post("/api/stt", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "transcript" in data
    assert "language" in data
    assert data["latency_ms"] >= 0.0


def test_api_stt_corrupted_file(test_client):
    files = {"file": ("corrupt.wav", io.BytesIO(b"INVALID_HEADER"), "audio/wav")}
    resp = test_client.post("/api/stt", files=files)
    assert resp.status_code == 400


def test_api_tts_endpoint(test_client):
    payload = {
        "text": "Based on the rules, students must maintain 75% attendance.",
        "voice_id": "telugu_female_1",
        "language": "te"
    }
    resp = test_client.post("/api/tts", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert "x-tts-engine" in resp.headers
    assert "x-tts-is-fallback" in resp.headers
    assert len(resp.content) > 500


def test_api_voice_chat_end_to_end(test_client, sample_pdf_bytes, sample_wav_bytes):
    from app.storage.qdrant import QdrantVectorStore
    if not QdrantVectorStore().is_connected():
        pytest.skip("Qdrant container offline; run 'docker compose up -d' for live integration tests.")

    # Ingest document first
    files_pdf = {"file": ("rules.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    test_client.post("/documents/upload", files=files_pdf)


    # 1. Turn 1: Voice Chat without conversation_id
    files_audio = {"file": ("query.wav", io.BytesIO(sample_wav_bytes), "audio/wav")}
    resp1 = test_client.post("/api/voice-chat", files=files_audio)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert "query_transcript" in data1
    assert "conversation_id" in data1
    assert "answer" in data1
    assert "audio_base64" in data1
    assert "stt_engine" in data1
    assert "tts_engine" in data1
    assert "is_fallback" in data1
    assert len(data1["audio_base64"]) > 100
    assert "latency" in data1
    assert data1["latency"]["total_ms"] > 0

    conv_id = data1["conversation_id"]

    # 2. Turn 2: Voice Chat with existing conversation_id
    files_audio_turn2 = {"file": ("query2.wav", io.BytesIO(sample_wav_bytes), "audio/wav")}
    resp2 = test_client.post(f"/api/voice-chat?conversation_id={conv_id}", files=files_audio_turn2)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["conversation_id"] == conv_id

    # 3. Verify PostgreSQL Conversation History has both turns (4 messages total)
    hist_resp = test_client.get(f"/conversations/{conv_id}")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["total_messages"] == 4
