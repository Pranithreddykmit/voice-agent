"""
Pytest configuration and shared test fixtures.
"""

import sys
import os
import uuid
from typing import List
import pytest
from fastapi.testclient import TestClient
import pymupdf

# Ensure ai-service is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-service")))

from app.config import settings
from app.embeddings.provider import BaseEmbeddingProvider
from app.generation.llm import BaseLLMClient
from app.rag import RAGPipeline
from app.storage.qdrant import QdrantVectorStore
from app.main import app, get_pipeline


class DeterministicMockEmbeddingProvider(BaseEmbeddingProvider):
    """Generates deterministic mock vectors for fast, offline testing."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            results.append(self.embed_query(text))
        return results

    def embed_query(self, text: str) -> List[float]:
        # Hash text to generate a deterministic unit vector
        val = sum(ord(c) for c in text) % 1000
        vec = [float((val + i) % 10) / 10.0 for i in range(self._dim)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class DeterministicMockLLMClient(BaseLLMClient):
    """Deterministic LLM client for tests."""

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 100
    ) -> str:
        if "No relevant document chunks" in prompt:
            return "The provided document does not contain information to answer this question."
        elif "attendance" in prompt.lower() or "హాజరు" in prompt:
            return "According to the academic regulations (Page 1), the minimum attendance requirement is 75%."
        elif "fee" in prompt.lower() or "ఫీజు" in prompt:
            return "According to Section 3 (Page 3), the standard registration deadline is October 15, 2026."
        return "Grounded test response based on provided context."


@pytest.fixture
def mock_embedding_provider():
    return DeterministicMockEmbeddingProvider(dim=64)


@pytest.fixture
def mock_llm_client():
    return DeterministicMockLLMClient()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Attendance Rule: Minimum 75% attendance is required.", fontsize=12)
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Grading Rule: Grade A requires 80% or above.", fontsize=12)
    pdf_data = doc.tobytes()
    doc.close()
    return pdf_data


@pytest.fixture
def sample_empty_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    doc.new_page()  # Page with no text
    pdf_data = doc.tobytes()
    doc.close()
    return pdf_data


@pytest.fixture
def test_client(mock_embedding_provider, mock_llm_client):
    """TestClient with mocked pipeline dependencies and isolated test collection."""
    test_collection = f"test_api_{uuid.uuid4().hex[:8]}"
    test_store = QdrantVectorStore(collection_name=test_collection)
    test_pipeline = RAGPipeline(
        embedding_provider=mock_embedding_provider,
        vector_store=test_store,
        llm_client=mock_llm_client
    )

    from app.main import get_pipeline, get_stt, get_tts
    from app.speech.stt import MockSTTProvider
    from app.speech.tts import FallbackTTSProvider

    app.dependency_overrides[get_pipeline] = lambda: test_pipeline
    app.dependency_overrides[get_stt] = lambda: MockSTTProvider()
    app.dependency_overrides[get_tts] = lambda: FallbackTTSProvider(sample_rate=16000)

    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()

    try:
        if test_store.is_connected():
            test_store.client.delete_collection(test_collection)
    except Exception:
        pass
