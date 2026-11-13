"""
Reasoning and Response Synthesis Node.
Generates strictly grounded answers for knowledge queries, explains tool calculations,
and responds naturally to casual or ambiguous user inputs.
"""

import time
import logging
from typing import Dict, Any, List

from app.agents.state import AgentState
from app.agents.tools.calculator import evaluate_arithmetic
from app.generation.llm import BaseLLMClient
from app.generation.prompts import format_context_prompt

logger = logging.getLogger(__name__)


def synthesis_node(state: AgentState, llm_client: BaseLLMClient) -> AgentState:
    """
    LangGraph Node: Performs reasoning and draft answer synthesis.
    """
    t0 = time.perf_counter()
    intent = state.get("intent", "knowledge_query")
    query = state.get("query", "")
    lang = state.get("language", "en")
    chunks = state.get("retrieved_chunks", [])
    history = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    validation_reason = state.get("validation_reason")

    tool_results = None
    draft_answer = ""

    # --------------------------------------------------------------------------
    # 1. Tool Execution Path
    # --------------------------------------------------------------------------
    if intent == "tool_query":
        t_tool_0 = time.perf_counter()
        tool_results = evaluate_arithmetic(query)
        tool_ms = (time.perf_counter() - t_tool_0) * 1000.0

        timings = state.get("timings", {})
        timings["tool_ms"] = round(tool_ms, 2)

        if tool_results["status"] == "success":
            res_val = tool_results["formatted_result"]
            expr = tool_results["expression"]
            if lang == "te":
                draft_answer = f"గణిత లెక్కింపు ప్రకారం, {expr} విలువ: **{res_val}**."
            else:
                draft_answer = f"According to the calculation, {expr} = **{res_val}**."
        else:
            err = tool_results["error"]
            if lang == "te":
                draft_answer = f"లెక్కింపులో లోపం ఏర్పడింది: {err}"
            else:
                draft_answer = f"Calculation error: {err}"

    # --------------------------------------------------------------------------
    # 2. Casual Small-Talk Path
    # --------------------------------------------------------------------------
    elif intent == "casual_query":
        q_lower = query.lower()
        if lang == "te" or "నమస్కారం" in q_lower or "హలో" in q_lower:
            draft_answer = "నమస్కారం! నేను మీ తెలుగు మల్టీ-ఏజెంట్ AI అసిస్టెంట్‌ని. మీకు విశ్వవిద్యాలయ నిబంధనలు లేదా విద్యా సమాచారంలో ఎలా సహాయపడగలను?"
        elif "thank" in q_lower or "ధన్యవాదాలు" in q_lower:
            draft_answer = "You're welcome! Feel free to ask if you have any more questions about academic regulations."
        else:
            draft_answer = "Hello! I am your Multi-Agent Telugu Conversational AI assistant. How can I assist you with academic regulations and document queries today?"

    # --------------------------------------------------------------------------
    # 3. Ambiguous Query / Clarification Path
    # --------------------------------------------------------------------------
    elif intent == "ambiguous_query":
        if lang == "te":
            draft_answer = "దయచేసి మీ ప్రశ్నను మరింత వివరంగా అడగండి. మీరు ఏ అంశం లేదా నిబంధన గురించి తెలుసుకోవాలనుకుంటున్నారు?"
        else:
            draft_answer = "Could you please clarify your question? Please specify which section or regulation you would like to know about."

    # --------------------------------------------------------------------------
    # 4. Grounded Knowledge Query Path
    # --------------------------------------------------------------------------
    else:
        # Build prompt with retrieved context
        user_prompt = format_context_prompt(query=query, retrieved_chunks=chunks)

        # If retrying due to validation failure, append corrective instruction
        if retry_count > 0 and validation_reason:
            user_prompt += f"\n\n[Correction Notice: Previous attempt was rejected because: {validation_reason}. Ensure strict grounding and exact citations.]"

        try:
            draft_answer = llm_client.generate(prompt=user_prompt)
        except Exception as gen_err:
            logger.warning(f"LLM generation failed ({gen_err}); falling back to direct context extraction.")
            if chunks:
                first_chunk = chunks[0]
                top_text = getattr(first_chunk, "text", "") if hasattr(first_chunk, "text") else (first_chunk.get("text", "") if isinstance(first_chunk, dict) else str(first_chunk))
                draft_answer = f"పత్రాల ప్రకారం: {top_text}" if lang == "te" else f"According to the documents: {top_text}"
            elif lang == "te":
                draft_answer = "మీరు అడిగిన ప్రశ్నకు సంబంధించిన సమాచారం పత్రాలలో లభించలేదు."
            else:
                draft_answer = "I could not find relevant information in the uploaded documents."

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    timings = state.get("timings", {})
    timings["generation_ms"] = round(elapsed_ms, 2)

    logger.info(f"[Synthesis Node] Intent: {intent} | Generated draft answer ({elapsed_ms:.2f}ms)")

    return {
        **state,
        "draft_answer": draft_answer,
        "tool_results": tool_results,
        "timings": timings
    }
