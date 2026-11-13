"""
LangGraph Shared State Definition for Multi-Agent Workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional
from app.schemas import RetrievedChunk, SourceCitation


class AgentState(TypedDict, total=False):
    """
    Typed dictionary representing the shared execution state of the multi-agent graph.
    """
    conversation_id: str
    query: str
    language: str  # "en", "te", "mixed"
    intent: str    # "knowledge_query", "casual_query", "tool_query", "ambiguous_query"
    needs_retrieval: bool
    needs_clarification: bool
    
    # Context window & Memory
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, ...]

    # Retrieval parameters and artifacts
    retrieval_mode: str
    top_k: Optional[int]
    document_id: Optional[str]
    min_score: Optional[float]
    retrieved_chunks: List[RetrievedChunk]

    # Tool Execution
    tool_results: Optional[Dict[str, Any]]

    # Reasoning & Synthesis
    draft_answer: str
    citations: List[SourceCitation]

    # Validation & Bounded Retries
    validation_result: bool
    validation_reason: Optional[str]
    retry_count: int
    max_retries: int

    # Final Output
    final_answer: str

    # Latency tracking
    timings: Dict[str, float]
