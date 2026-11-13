"""
Integration and end-to-end tests for RAGPipeline.
"""

import pytest
import uuid
from app.rag import RAGPipeline
from app.storage.qdrant import QdrantVectorStore
from app.schemas import QueryRequest


@pytest.fixture
def isolated_pipeline(mock_embedding_provider, mock_llm_client):
    test_collection = f"test_rag_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(collection_name=test_collection)
    if not store.is_connected():
        pytest.skip("Qdrant container offline; run 'docker compose up -d' for live integration tests.")
    pipeline = RAGPipeline(
        embedding_provider=mock_embedding_provider,
        vector_store=store,
        llm_client=mock_llm_client
    )
    yield pipeline
    try:
        if store.is_connected():
            store.client.delete_collection(test_collection)
    except Exception:
        pass


def test_rag_pipeline_end_to_end(isolated_pipeline, sample_pdf_bytes):
    # 1. Ingest PDF
    upload_res = isolated_pipeline.ingest_pdf(
        file_bytes=sample_pdf_bytes,
        filename="regulations_2026.pdf"
    )
    assert upload_res.status == "success"
    assert upload_res.total_pages == 2
    assert upload_res.total_chunks >= 2
    assert upload_res.processing_time_ms > 0

    # 2. Query for attendance
    query_req = QueryRequest(query="What is the minimum attendance requirement?")
    query_res = isolated_pipeline.query(query_req)

    assert query_res.query == "What is the minimum attendance requirement?"
    assert "75%" in query_res.answer
    assert len(query_res.sources) >= 1
    assert query_res.sources[0].document == "regulations_2026.pdf"
    assert query_res.sources[0].page in [1, 2]
    assert query_res.latency.total_ms > 0
    assert query_res.latency.retrieval_ms > 0
    assert query_res.latency.generation_ms >= 0


def test_rag_pipeline_out_of_context(isolated_pipeline):
    # Query empty database with no chunks
    query_req = QueryRequest(query="What is the chemical formula for photosynthesis?")
    query_res = isolated_pipeline.query(query_req)

    assert "does not contain" in query_res.answer.lower()
    assert len(query_res.sources) == 0
