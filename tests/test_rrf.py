"""
Unit tests for Reciprocal Rank Fusion (RRF).
"""

import pytest
from app.retrieval.rrf import reciprocal_rank_fusion
from app.schemas import RetrievedChunk


def test_rrf_basic_fusion():
    chunk_a = RetrievedChunk(
        chunk_id="cA", document_id="d1", source="s.pdf", page=1, chunk_index=0,
        text="Text A", score=0.9, retrieval_method="dense", rank=1
    )
    chunk_b = RetrievedChunk(
        chunk_id="cB", document_id="d1", source="s.pdf", page=2, chunk_index=1,
        text="Text B", score=0.8, retrieval_method="dense", rank=2
    )
    chunk_c = RetrievedChunk(
        chunk_id="cC", document_id="d1", source="s.pdf", page=3, chunk_index=2,
        text="Text C", score=12.0, retrieval_method="bm25", rank=1
    )

    # cA is rank 1 in dense, rank 2 in bm25
    dense_list = [chunk_a, chunk_b]
    bm25_list = [chunk_c, chunk_a]

    fused = reciprocal_rank_fusion(
        ranked_lists={"dense": dense_list, "bm25": bm25_list},
        k=60,
        top_k=3
    )

    assert len(fused) == 3
    # cA should be top ranked because it appears in both lists: (1/61 + 1/62) > 1/61
    assert fused[0].chunk_id == "cA"
    assert "dense+bm25" in fused[0].retrieval_method
    assert fused[0].rank == 1

    expected_score_a = round((1.0 / 61.0) + (1.0 / 62.0), 6)
    assert fused[0].score == expected_score_a


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []
