"""
Unit tests for Query Understanding and Intent Classification Node.
"""

import pytest
from app.agents.state import AgentState
from app.agents.nodes.understanding import (
    detect_language,
    classify_intent,
    query_understanding_node
)


def test_language_detection():
    assert detect_language("What is the attendance rule?") == "en"
    assert detect_language("కనీస హాజరు ఎంత ఉండాలి?") == "te"
    assert detect_language("revaluation fee ఎంత?") == "mixed"


def test_intent_classification_knowledge():
    intent, needs_ret, needs_clar = classify_intent("What is the minimum attendance requirement?", [])
    assert intent == "knowledge_query"
    assert needs_ret is True
    assert needs_clar is False

    intent_te, needs_ret_te, _ = classify_intent("పరీక్షలకు కనీసం ఎంత శాతం హాజరు ఉండాలి?", [])
    assert intent_te == "knowledge_query"
    assert needs_ret_te is True


def test_intent_classification_tool():
    intent, needs_ret, _ = classify_intent("What is 15% of 240?", [])
    assert intent == "tool_query"
    assert needs_ret is False

    intent_math, needs_ret_math, _ = classify_intent("Calculate 125 * 18", [])
    assert intent_math == "tool_query"
    assert needs_ret_math is False


def test_intent_classification_casual():
    intent, needs_ret, _ = classify_intent("Hello, how are you?", [])
    assert intent == "casual_query"
    assert needs_ret is False

    intent_te, needs_ret_te, _ = classify_intent("నమస్కారం", [])
    assert intent_te == "casual_query"
    assert needs_ret_te is False


def test_intent_classification_ambiguous_and_followup():
    # Without prior history: ambiguous -> clarification
    intent_no_hist, _, needs_clar = classify_intent("What about the second one?", [])
    assert intent_no_hist == "ambiguous_query"
    assert needs_clar is True

    # With prior history: resolved as contextual knowledge query
    history = [{"role": "user", "content": "Tell me about Section 1 and Section 2"}]
    intent_with_hist, needs_ret, needs_clar2 = classify_intent("What about the second one?", history)
    assert intent_with_hist == "knowledge_query"
    assert needs_ret is True
    assert needs_clar2 is False


def test_query_understanding_node():
    state: AgentState = {
        "query": "15% of 240",
        "messages": [],
        "timings": {}
    }
    updated = query_understanding_node(state)
    assert updated["intent"] == "tool_query"
    assert updated["needs_retrieval"] is False
    assert "understanding_ms" in updated["timings"]
