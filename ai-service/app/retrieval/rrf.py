"""
Reciprocal Rank Fusion (RRF) Implementation.
Combines multiple ranked retrieval result lists into a single unified ranking.

Formula:
    RRF_score(d) = Σ (weight_m / (k + rank_m(d)))
where:
    k = smoothing constant (default: 60)
    rank_m(d) = 1-indexed position of document d in list m
    weight_m = optional weighting multiplier for method m (default: 1.0)
"""

import logging
from typing import List, Dict, Optional, Tuple
from app.config import settings
from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[RetrievedChunk]],
    k: int = 60,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None
) -> List[RetrievedChunk]:
    """
    Combines ranked results from different retrieval methods using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Dictionary mapping method name (e.g. 'dense', 'bm25') to a ranked list of RetrievedChunk.
        k: Smoothing constant to prevent top-ranked outliers from dominating (default: 60).
        top_k: Maximum number of fused items to return.
        weights: Optional dictionary of method weights (e.g. {'dense': 1.0, 'bm25': 1.0}).

    Returns:
        List of fused RetrievedChunk objects sorted by combined RRF score in descending order.
    """
    if not ranked_lists:
        return []

    weights = weights or {}
    rrf_scores: Dict[str, float] = {}
    chunk_store: Dict[str, RetrievedChunk] = {}
    contributing_methods: Dict[str, List[str]] = {}

    for method, chunk_list in ranked_lists.items():
        weight = weights.get(method, 1.0)
        for rank, chunk in enumerate(chunk_list, start=1):
            chunk_id = chunk.chunk_id
            score_delta = weight / (k + rank)

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_delta

            if chunk_id not in chunk_store:
                chunk_store[chunk_id] = chunk
                contributing_methods[chunk_id] = [method]
            else:
                if method not in contributing_methods[chunk_id]:
                    contributing_methods[chunk_id].append(method)

    # Sort chunks by fused RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_results: List[RetrievedChunk] = []
    for rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
        base_chunk = chunk_store[cid]
        methods_str = "+".join(contributing_methods[cid])
        fused_chunk = RetrievedChunk(
            chunk_id=base_chunk.chunk_id,
            document_id=base_chunk.document_id,
            source=base_chunk.source,
            page=base_chunk.page,
            chunk_index=base_chunk.chunk_index,
            text=base_chunk.text,
            score=float(round(rrf_scores[cid], 6)),
            retrieval_method=f"rrf({methods_str})",
            rank=rank
        )
        fused_results.append(fused_chunk)

    logger.info(f"RRF fused {len(chunk_store)} unique chunks from {list(ranked_lists.keys())} into top {len(fused_results)}.")
    return fused_results
