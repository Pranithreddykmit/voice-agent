"""
Integration tests for QdrantVectorStore.
"""

import pytest
import uuid
from app.storage.qdrant import QdrantVectorStore
from app.schemas import DocumentChunk


@pytest.fixture
def temp_qdrant_store():
    # Use a dedicated test collection
    test_collection = f"test_col_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(collection_name=test_collection)
    if not store.is_connected():
        pytest.skip("Qdrant container offline; run 'docker compose up -d' for live integration tests.")
    yield store
    # Cleanup collection after test if connected
    try:
        if store.is_connected():
            store.client.delete_collection(test_collection)
    except Exception:
        pass


def test_qdrant_connection():
    store = QdrantVectorStore()
    if not store.is_connected():
        pytest.skip("Qdrant container offline; run 'docker compose up -d' for live integration tests.")
    assert store.is_connected() is True


def test_qdrant_upsert_and_search(temp_qdrant_store, mock_embedding_provider):
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="doc_rules",
            source="rules.pdf",
            page=1,
            chunk_index=0,
            text="75% minimum attendance requirement.",
            char_count=36
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="doc_rules",
            source="rules.pdf",
            page=2,
            chunk_index=1,
            text="Grade O requires 90% and above.",
            char_count=32
        )
    ]

    vectors = mock_embedding_provider.embed_documents([c.text for c in chunks])
    temp_qdrant_store.upsert_chunks(chunks, vectors)

    # Search for attendance
    query_vec = mock_embedding_provider.embed_query("75% minimum attendance requirement.")
    results = temp_qdrant_store.search(query_vec, top_k=2)

    assert len(results) >= 1
    assert results[0].source == "rules.pdf"
