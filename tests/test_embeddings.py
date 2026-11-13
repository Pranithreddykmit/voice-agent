"""
Unit and regression tests for Embedding Providers.
"""

import pytest
import numpy as np
from app.embeddings.provider import (
    BaseEmbeddingProvider,
    FastEmbedProvider,
    get_embedding_provider
)


def test_mock_embedding_provider(mock_embedding_provider):
    texts = ["First passage in Telugu", "Second passage in English"]
    vectors = mock_embedding_provider.embed_documents(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == mock_embedding_provider.dimension

    query_vec = mock_embedding_provider.embed_query("Telugu query")
    assert len(query_vec) == mock_embedding_provider.dimension


def test_embedding_factory_default():
    provider = get_embedding_provider()
    assert isinstance(provider, BaseEmbeddingProvider)
    assert provider.dimension == 1024


def test_e5_prefix_handling():
    """Verify that E5 models automatically prepend 'passage: ' and 'query: '."""
    provider = FastEmbedProvider(model_name="intfloat/multilingual-e5-large")
    assert provider._is_e5 is True

    # Document text
    doc_text = "కనీస హాజరు 75 శాతం."
    formatted_doc = provider._prepare_document_text(doc_text)
    assert formatted_doc == "passage: కనీస హాజరు 75 శాతం."

    # Already prefixed document
    already_prefixed_doc = "passage: existing text"
    assert provider._prepare_document_text(already_prefixed_doc) == "passage: existing text"

    # Query text
    query_text = "కనీస హాజరు ఎంత?"
    formatted_query = provider._prepare_query_text(query_text)
    assert formatted_query == "query: కనీస హాజరు ఎంత?"

    # Already prefixed query
    already_prefixed_query = "query: existing query"
    assert provider._prepare_query_text(already_prefixed_query) == "query: existing query"


def test_real_fastembed_telugu_semantic_similarity():
    """Integration test verifying real multilingual-e5-large semantic similarity for Telugu."""
    provider = FastEmbedProvider(model_name="intfloat/multilingual-e5-large")
    assert provider.dimension == 1024

    doc_telugu = "విశ్వవిద్యాలయ నిబంధనల ప్రకారం సెమిస్టర్ పరీక్షలకు హాజరు కావడానికి కనీసం 75% హాజరు తప్పనిసరి."
    doc_unrelated = "కిరణజన్య సంయోగక్రియ ద్వారా మొక్కలు ఆహారాన్ని తయారుచేసుకుంటాయి."
    query_telugu = "పరీక్షలకు కనీస హాజరు ఎంత శాతం ఉండాలి?"

    doc_vectors = provider.embed_documents([doc_telugu, doc_unrelated])
    query_vector = provider.embed_query(query_telugu)

    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_relevant = cos_sim(query_vector, doc_vectors[0])
    sim_unrelated = cos_sim(query_vector, doc_vectors[1])

    # Relevant Telugu document must have significantly higher similarity than unrelated topic
    assert sim_relevant > 0.80
    assert sim_relevant > sim_unrelated + 0.10
