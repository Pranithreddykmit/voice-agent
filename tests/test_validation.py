"""
Unit tests for Response Validation Node.
"""

import pytest
from app.agents.state import AgentState
from app.agents.nodes.validation import validate_response, validation_node
from app.schemas import RetrievedChunk


def test_validation_tool_success():
    state: AgentState = {
        "intent": "tool_query",
        "draft_answer": "According to the calculation, 15% of 240 = **36**.",
        "tool_results": {"status": "success", "formatted_result": "36"},
        "retrieved_chunks": []
    }
    is_valid, reason = validate_response(state)
    assert is_valid is True


def test_validation_tool_inconsistent():
    state: AgentState = {
        "intent": "tool_query",
        "draft_answer": "According to the calculation, the result is 42.",
        "tool_results": {"status": "success", "formatted_result": "36"},
        "retrieved_chunks": []
    }
    is_valid, reason = validate_response(state)
    assert is_valid is False
    assert "36" in reason


def test_validation_knowledge_valid_citation():
    chunks = [
        RetrievedChunk(
            chunk_id="c1", document_id="d1", source="rules.pdf", page=1, chunk_index=0,
            text="Every student must maintain minimum 75% attendance.", score=0.9
        )
    ]
    state: AgentState = {
        "intent": "knowledge_query",
        "draft_answer": "Based on the rules (Source: rules.pdf, Page 1), students must have 75% attendance.",
        "retrieved_chunks": chunks
    }
    is_valid, reason = validate_response(state)
    assert is_valid is True


def test_validation_knowledge_hallucinated_page():
    chunks = [
        RetrievedChunk(
            chunk_id="c1", document_id="d1", source="rules.pdf", page=1, chunk_index=0,
            text="Every student must maintain minimum 75% attendance.", score=0.9
        )
    ]
    # Citing Page 9 which was never retrieved
    state: AgentState = {
        "intent": "knowledge_query",
        "draft_answer": "Based on the rules (Source: rules.pdf, Page 9), students must have 75% attendance.",
        "retrieved_chunks": chunks
    }
    is_valid, reason = validate_response(state)
    assert is_valid is False
    assert "Page" in reason or "page" in reason


def test_validation_casual_clean():
    state: AgentState = {
        "intent": "casual_query",
        "draft_answer": "Hello! How can I assist you with your document queries?",
        "retrieved_chunks": []
    }
    is_valid, reason = validate_response(state)
    assert is_valid is True


def test_validation_refusal():
    state: AgentState = {
        "intent": "knowledge_query",
        "draft_answer": "The provided document does not contain information to answer this question.",
        "retrieved_chunks": []
    }
    is_valid, reason = validate_response(state)
    assert is_valid is True
