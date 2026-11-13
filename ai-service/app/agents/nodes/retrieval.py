"""
Knowledge Retrieval Node for LangGraph Orchestrator.
Delegates search to the Phase 2 Hybrid RAG Pipeline without duplicating retrieval code.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from app.agents.state import AgentState
from app.config import settings
from app.rag import RAGPipeline
from app.schemas import SourceCitation, RetrievedChunk

logger = logging.getLogger(__name__)


def retrieval_node(state: AgentState, rag_pipeline: RAGPipeline) -> AgentState:
    """
    LangGraph Node: Invokes the Phase 2 multi-mode retrieval subsystem.
    """
    t0 = time.perf_counter()
    query = state.get("query", "")
    retrieval_mode = state.get("retrieval_mode") or settings.RETRIEVAL_MODE
    top_k = state.get("top_k") or settings.RETRIEVAL_TOP_K
    document_id = state.get("document_id")
    min_score = state.get("min_score")

    # If contextual history exists, enrich query representation if needed
    context_msgs = state.get("messages", [])
    search_query = query
    if context_msgs and len(query.split()) < 6:
        # Prepend last user query topic if follow-up is very short
        last_user_msg = next((m["content"] for m in reversed(context_msgs) if m.get("role") == "user"), None)
        if last_user_msg and last_user_msg != query:
            search_query = f"{last_user_msg} {query}"

    retrieved_chunks, sub_timings = rag_pipeline.retrieve(
        query=search_query,
        mode=retrieval_mode,
        top_k=top_k,
        document_id=document_id,
        min_score=min_score
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Build citations
    citations: List[SourceCitation] = [
        SourceCitation(
            document=chunk.source,
            page=chunk.page,
            score=round(chunk.score, 4),
            chunk_id=chunk.chunk_id,
            snippet=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
            retrieval_method=chunk.retrieval_method,
            rank=chunk.rank
        )
        for chunk in retrieved_chunks
    ]

    timings = state.get("timings", {})
    timings["retrieval_ms"] = round(elapsed_ms, 2)
    timings["dense_ms"] = round(sub_timings.get("dense_ms", 0.0), 2)
    timings["bm25_ms"] = round(sub_timings.get("bm25_ms", 0.0), 2)
    timings["rrf_ms"] = round(sub_timings.get("rrf_ms", 0.0), 2)
    timings["rerank_ms"] = round(sub_timings.get("rerank_ms", 0.0), 2)

    logger.info(
        f"[Retrieval Node] Mode: {retrieval_mode} | Retrieved: {len(retrieved_chunks)} chunks ({elapsed_ms:.2f}ms)"
    )

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "citations": citations,
        "timings": timings
    }
