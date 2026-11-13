"""
Unit tests for Cross-Encoder Rerankers.
"""

import pytest
from app.retrieval.reranker import MockReranker, BaseReranker, get_reranker
from app.schemas import RetrievedChunk


def test_mock_reranker_reordering():
    reranker = MockReranker()
    candidates = [
        RetrievedChunk(
            chunk_id="c1", document_id="d1", source="doc.pdf", page=2, chunk_index=0,
            text="Unrelated grading text about Grade O and Grade A.", score=0.8, rank=1
        ),
        RetrievedChunk(
            chunk_id="c2", document_id="d1", source="doc.pdf", page=1, chunk_index=1,
            text="Attendance clause: Minimum 75% attendance is required for exams.", score=0.7, rank=2
        )
    ]

    reranked = reranker.rerank(query="minimum attendance", candidates=candidates, top_k=2)
    assert len(reranked) == 2
    # c2 contains both "minimum" and "attendance", so it should be promoted to rank 1
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rank == 1
    assert "rerank" in reranked[0].retrieval_method


def test_reranker_empty_candidates():
    reranker = MockReranker()
    assert reranker.rerank(query="test", candidates=[], top_k=5) == []


def test_reranker_factory():
    reranker = get_reranker()
    assert isinstance(reranker, BaseReranker)
