"""
API endpoint tests for Phase 3 LangGraph Multi-Agent System and Conversational Memory.
"""

import io
import pytest
from app.storage.qdrant import QdrantVectorStore


def test_api_query_generates_conversation_id(test_client, sample_pdf_bytes):
    if not QdrantVectorStore().is_connected():
        pytest.skip("Qdrant container offline; run 'docker compose up -d' for live integration tests.")
    # Ingest PDF
    files = {"file": ("rules.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    test_client.post("/documents/upload", files=files)

    # Query without conversation_id
    payload = {"query": "What is the minimum attendance requirement?"}
    resp = test_client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "conversation_id" in data
    assert len(data["conversation_id"]) > 0
    assert data["intent"] == "knowledge_query"
    assert len(data["sources"]) >= 1
    assert data["validation"]["valid"] is True


def test_api_multi_turn_conversation_and_history(test_client):
    # Turn 1: Casual greeting
    turn1_resp = test_client.post("/query", json={"query": "Hello!"})
    assert turn1_resp.status_code == 200
    turn1_data = turn1_resp.json()
    conv_id = turn1_data["conversation_id"]
    assert turn1_data["intent"] == "casual_query"
    assert len(turn1_data["sources"]) == 0

    # Turn 2: Calculation using same conversation ID
    turn2_resp = test_client.post("/query", json={"conversation_id": conv_id, "query": "What is 15% of 240?"})
    assert turn2_resp.status_code == 200
    turn2_data = turn2_resp.json()
    assert turn2_data["conversation_id"] == conv_id
    assert turn2_data["intent"] == "tool_query"
    assert turn2_data["tool_result"]["result"] == 36

    # Turn 3: Get Conversation History
    hist_resp = test_client.get(f"/conversations/{conv_id}")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["conversation_id"] == conv_id
    assert hist_data["total_messages"] == 4  # 2 user messages + 2 assistant messages
    assert hist_data["messages"][0]["role"] == "user"
    assert hist_data["messages"][0]["content"] == "Hello!"
    assert hist_data["messages"][1]["role"] == "assistant"
    assert hist_data["messages"][2]["role"] == "user"
    assert hist_data["messages"][2]["content"] == "What is 15% of 240?"
