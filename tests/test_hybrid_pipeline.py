"""
Integration tests for Multi-Mode Hybrid Retrieval and Reranking Pipeline.
"""

import io
import uuid
import pytest
from app.rag import RAGPipeline
from app.storage.qdrant import QdrantVectorStore
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import MockReranker
from app.schemas import QueryRequest


@pytest.fixture
def multi_mode_pipeline(mock_embedding_provider, mock_llm_client):
    test_collection = f"test_hybrid_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(collection_name=test_collection)
    if not store.is_connected():
        pytest.skip("Qdrant container offline; run 'docker compose up -d' for live integration tests.")
    bm25 = BM25Index()
    reranker = MockReranker()

    pipeline = RAGPipeline(
        embedding_provider=mock_embedding_provider,
        vector_store=store,
        bm25_index=bm25,
        reranker=reranker,
        llm_client=mock_llm_client
    )
    yield pipeline

    try:
        if store.is_connected():
            store.client.delete_collection(test_collection)
    except Exception:
        pass


def test_all_retrieval_modes(multi_mode_pipeline, sample_pdf_bytes):
    # Ingest document
    upload_res = multi_mode_pipeline.ingest_pdf(sample_pdf_bytes, filename="academic_rules.pdf")
    assert upload_res.status == "success"
    assert multi_mode_pipeline.bm25_index.is_indexed is True

    # 1. Test Dense Mode
    res_dense, timings_dense = multi_mode_pipeline.retrieve(
        query="minimum attendance", mode="dense", top_k=2
    )
    assert len(res_dense) >= 1
    assert res_dense[0].retrieval_method == "dense"
    assert timings_dense["dense_ms"] > 0

    # 2. Test BM25 Mode
    res_bm25, timings_bm25 = multi_mode_pipeline.retrieve(
        query="minimum attendance", mode="bm25", top_k=2
    )
    assert len(res_bm25) >= 1
    assert res_bm25[0].retrieval_method == "bm25"
    assert timings_bm25["bm25_ms"] >= 0

    # 3. Test Hybrid Mode (RRF)
    res_hybrid, timings_hybrid = multi_mode_pipeline.retrieve(
        query="minimum attendance", mode="hybrid", top_k=2
    )
    assert len(res_hybrid) >= 1
    assert "rrf" in res_hybrid[0].retrieval_method
    assert timings_hybrid["rrf_ms"] >= 0

    # 4. Test Hybrid Rerank Mode
    res_rerank, timings_rerank = multi_mode_pipeline.retrieve(
        query="minimum attendance", mode="hybrid_rerank", top_k=2
    )
    assert len(res_rerank) >= 1
    assert "rerank" in res_rerank[0].retrieval_method
    assert timings_rerank["rerank_ms"] >= 0


def test_query_api_with_retrieval_mode(test_client, sample_pdf_bytes):
    if not QdrantVectorStore().is_connected():
        pytest.skip("Qdrant container offline; run 'docker compose up -d' for live integration tests.")

    # Ingest PDF
    files = {"file": ("rules.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    test_client.post("/documents/upload", files=files)

    # Query with hybrid_rerank mode
    payload = {
        "query": "What is the attendance rule?",
        "retrieval_mode": "hybrid_rerank",
        "top_k": 2
    }
    resp = test_client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieval_mode"] == "hybrid_rerank"
    assert len(data["sources"]) >= 1
    assert "latency" in data
    assert data["latency"]["total_ms"] > 0
