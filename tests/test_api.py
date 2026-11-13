"""
API endpoint tests using FastAPI TestClient.
"""

import io
import pytest
from app.storage.qdrant import QdrantVectorStore


def test_api_health(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert "qdrant_connected" in data
    assert "embedding_provider" in data
    assert "llm_provider" in data


def test_api_upload_pdf(test_client, sample_pdf_bytes):
    if not QdrantVectorStore().is_connected():
        pytest.skip("Qdrant container offline; run 'docker compose up -d' for live integration tests.")
    files = {
        "file": ("academic_rules.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    response = test_client.post("/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "academic_rules.pdf"
    assert data["total_pages"] == 2
    assert data["total_chunks"] >= 2
    assert data["status"] == "success"


def test_api_upload_invalid_file_extension(test_client):
    files = {
        "file": ("notes.txt", io.BytesIO(b"Some text"), "text/plain")
    }
    response = test_client.post("/documents/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files are accepted" in response.json()["detail"]


def test_api_query(test_client, sample_pdf_bytes):
    if not QdrantVectorStore().is_connected():
        pytest.skip("Qdrant container offline; run 'docker compose up -d' for live integration tests.")
    # Ingest document first
    files = {
        "file": ("academic_rules.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    test_client.post("/documents/upload", files=files)

    # Query endpoint
    query_payload = {
        "query": "What is the minimum attendance?",
        "top_k": 3
    }
    response = test_client.post("/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency" in data
    assert data["latency"]["total_ms"] > 0


def test_api_query_invalid_payload(test_client):
    response = test_client.post("/query", json={"query": ""})
    assert response.status_code == 422
