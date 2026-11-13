"""
Query Understanding & Intent Classification Node.
Determines language, classifies intent, and performs contextual disambiguation using conversation history.
"""

import re
import time
import logging
from typing import Dict, Any, Tuple
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

# Patterns for intent detection
TELUGU_PATTERN = re.compile(r"[\u0C00-\u0C7F]", re.UNICODE)
LATIN_PATTERN = re.compile(r"[a-zA-Z]", re.UNICODE)

CASUAL_GREETINGS = {
    "hello", "hi", "hey", "good morning", "good evening", "how are you",
    "who are you", "what can you do", "thanks", "thank you", "bye", "goodbye",
    "నమస్కారం", "హలో", "ధన్యవాదాలు", "ఎలా ఉన్నారు", "బాగున్నారా"
}

CALC_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%\s*(?:of|లో)?\s*\d+)|"
    r"(?:\d+\s*[\+\-\*\/\^%]\s*\d+)|"
    r"(?:^(?:calculate|compute|find|eval|లెక్కింపు)\s+[\d\s\+\-\*\/\(\)\.%]+)",
    re.IGNORECASE
)

AMBIGUOUS_PATTERNS = [
    r"^(?:what about|how about|tell me about)\s+(?:the\s+)?(?:second|third|other|next|previous|it|that|this)(?:\s+one)?\s*$",
    r"^(?:explain|clarify)\s+(?:it|that|this)\s*$",
    r"^(?:and then|what else)\s*$"
]


def detect_language(text: str) -> str:
    """Detects if query is English ('en'), Telugu ('te'), or mixed ('mixed')."""
    has_telugu = bool(TELUGU_PATTERN.search(text))
    has_latin = bool(LATIN_PATTERN.search(text))

    if has_telugu and has_latin:
        return "mixed"
    elif has_telugu:
        return "te"
    else:
        return "en"


def classify_intent(query: str, history: list) -> Tuple[str, bool, bool]:
    """
    Classifies intent into:
    - 'tool_query' (math/calculation)
    - 'casual_query' (greetings, small talk)
    - 'ambiguous_query' (needs clarification when context is missing)
    - 'knowledge_query' (document / factual query)

    Returns: (intent, needs_retrieval, needs_clarification)
    """
    q_clean = query.strip().lower().rstrip("?।.")

    # 1. Check Tool / Math Query
    if CALC_PATTERN.search(query) or any(w in q_clean.split() for w in ["calculate", "compute", "లెక్కింపు"]):
        return "tool_query", False, False

    # 2. Check Casual Greeting / Small Talk
    if q_clean in CASUAL_GREETINGS or any(q_clean.startswith(g) for g in ["hello", "hi ", "hey", "నమస్కారం"]):
        # But if it also contains questions about attendance or regulations, treat as knowledge
        if not any(k in q_clean for k in ["attendance", "exam", "grade", "fee", "regulation", "హాజరు", "పరీక్ష", "ఫీజు", "నిబంధన"]):
            return "casual_query", False, False

    # 3. Check Ambiguity / Follow-up without prior history
    is_ambiguous_shape = any(re.search(pat, q_clean) for pat in AMBIGUOUS_PATTERNS)
    if is_ambiguous_shape:
        if not history:
            return "ambiguous_query", False, True
        else:
            # Context exists, resolve as knowledge follow-up
            return "knowledge_query", True, False

    # Default to Knowledge Query
    return "knowledge_query", True, False


def query_understanding_node(state: AgentState) -> AgentState:
    """
    LangGraph Node: Analyzes the query, identifies language and intent,
    and updates routing flags in state.
    """
    t0 = time.perf_counter()
    query = state.get("query", "")
    history = state.get("messages", [])

    lang = detect_language(query)
    intent, needs_retrieval, needs_clarification = classify_intent(query, history)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    timings = state.get("timings", {})
    timings["understanding_ms"] = round(elapsed_ms, 2)

    logger.info(
        f"[Query Understanding] Query: '{query[:50]}...' | "
        f"Lang: {lang} | Intent: {intent} | Needs Retrieval: {needs_retrieval} ({elapsed_ms:.2f}ms)"
    )

    return {
        **state,
        "language": lang,
        "intent": intent,
        "needs_retrieval": needs_retrieval,
        "needs_clarification": needs_clarification,
        "timings": timings
    }
