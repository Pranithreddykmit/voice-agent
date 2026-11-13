"""
Response Validation Node for LangGraph Workflow.
Performs deterministic and semantic verification on citations, fact alignment, and tool outputs.
"""

import re
import time
import logging
from typing import Dict, Any, Tuple

from app.agents.state import AgentState

logger = logging.getLogger(__name__)


def validate_response(state: AgentState) -> Tuple[bool, str]:
    """
    Validates draft response according to intent-specific safety rules.

    Returns:
        (is_valid: bool, reason: str)
    """
    intent = state.get("intent", "knowledge_query")
    draft = state.get("draft_answer", "").strip()
    tool_res = state.get("tool_results")
    chunks = state.get("retrieved_chunks", [])

    if not draft:
        return False, "Generated draft answer is empty."

    # 1. Validation for Tool Queries
    if intent == "tool_query":
        if not tool_res:
            return False, "Tool execution result missing in state."
        if tool_res.get("status") == "success":
            expected_num = str(tool_res.get("formatted_result", ""))
            if expected_num and expected_num not in draft:
                return False, f"Draft answer does not contain the calculated value '{expected_num}'."
        return True, "Tool calculation validated successfully."

    # 2. Validation for Casual & Ambiguous Queries
    if intent in ["casual_query", "ambiguous_query"]:
        # Ensure no hallucinated document citations in small talk
        if "Page " in draft or "Source: " in draft:
            return False, "Casual small-talk response should not contain document citations."
        return True, "Casual / clarification response validated."

    # 3. Validation for Knowledge Queries
    is_refusal = (
        "does not contain" in draft.lower() or
        "సమాచారం లేదు" in draft or
        "not contain information" in draft.lower() or
        "insufficient" in draft.lower()
    )

    if not chunks:
        if is_refusal:
            return True, "Validated grounded refusal for absent/out-of-corpus context."
        else:
            return False, "Answer asserts factual claims when no document context was retrieved."

    if is_refusal:
        return True, "Validated grounded refusal."

    # Check that answer provides source citations
    has_citation = bool(
        re.search(r"(?i)(?:source|page|పుట|పేజీ)", draft) or
        any(c.source in draft for c in chunks)
    )

    if not has_citation and len(chunks) > 0:
        has_chunk_overlap = any(
            len(set(c.text.lower().split()).intersection(draft.lower().split())) >= 3
            for c in chunks
        )
        if not has_chunk_overlap:
            return False, "Draft answer does not cite sources or reflect retrieved document chunks."

    # Verify cited page numbers exist in retrieved chunks
    retrieved_pages = {c.page for c in chunks}
    cited_pages = {int(p) for p in re.findall(r"(?i)page\s*(\d+)", draft)}
    if cited_pages and not cited_pages.issubset(retrieved_pages):
        return False, f"Answer cited page(s) {cited_pages - retrieved_pages} not present in retrieved context."

    return True, "Grounded citations and content match retrieved document context."


def validation_node(state: AgentState) -> AgentState:
    """
    LangGraph Node: Validates the synthesized response and records audit info.
    """
    t0 = time.perf_counter()
    is_valid, reason = validate_response(state)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    timings = state.get("timings", {})
    timings["validation_ms"] = round(elapsed_ms, 2)

    logger.info(
        f"[Validation Node] Valid: {is_valid} | Reason: '{reason}' ({elapsed_ms:.2f}ms)"
    )

    return {
        **state,
        "validation_result": is_valid,
        "validation_reason": reason,
        "timings": timings
    }
