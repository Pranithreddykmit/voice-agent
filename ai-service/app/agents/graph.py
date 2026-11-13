"""
LangGraph Multi-Agent StateGraph Assembler and Orchestrator.
Coordinates query understanding, conditional routing, hybrid RAG retrieval,
safe tool execution, response synthesis, bounded validation retry loop, and fallback.
"""

import time
import logging
from typing import Dict, Any, Optional, Iterator

from langgraph.graph import StateGraph, END
from app.config import settings
from app.rag import RAGPipeline
from app.generation.llm import BaseLLMClient
from app.agents.state import AgentState
from app.agents.nodes.understanding import query_understanding_node
from app.agents.nodes.retrieval import retrieval_node
from app.agents.nodes.synthesis import synthesis_node
from app.agents.nodes.validation import validation_node
from app.agents.tools.calculator import evaluate_arithmetic
from app.generation.prompts import format_context_prompt

logger = logging.getLogger(__name__)


def build_agent_graph(rag_pipeline: RAGPipeline, llm_client: BaseLLMClient):
    """
    Constructs the compiled LangGraph StateGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # 1. Define Node Wrappers
    def node_understanding(state: AgentState) -> AgentState:
        return query_understanding_node(state)

    def node_retrieval(state: AgentState) -> AgentState:
        return retrieval_node(state, rag_pipeline=rag_pipeline)

    def node_synthesis(state: AgentState) -> AgentState:
        return synthesis_node(state, llm_client=llm_client)

    def node_validation(state: AgentState) -> AgentState:
        return validation_node(state)

    def node_retry(state: AgentState) -> AgentState:
        retries = state.get("retry_count", 0) + 1
        logger.warning(f"[Retry Handler] Validation failed. Initiating retry attempt {retries}/{state.get('max_retries', 2)}...")
        return {
            **state,
            "retry_count": retries
        }

    def node_fallback(state: AgentState) -> AgentState:
        lang = state.get("language", "en")
        reason = state.get("validation_reason", "Validation checks failed")
        logger.error(f"[Fallback Handler] Maximum retries exceeded ({reason}). Returning transparent refusal.")
        if lang == "te":
            fallback_text = "క్షమించండి, విశ్వసనీయ సమాచారంతో ధృవీకరించబడిన సమాధానం రూపొందించడం సాధ్యపడలేదు. దయచేసి మరింత స్పష్టమైన వివరాలతో మీ ప్రశ్నను మళ్లీ అడగండి."
        else:
            fallback_text = "I apologize, but I could not synthesize a sufficiently verified answer from the available document context. Please refine your query or consult official university documents directly."
        
        return {
            **state,
            "final_answer": fallback_text
        }

    def node_finalize(state: AgentState) -> AgentState:
        return {
            **state,
            "final_answer": state.get("draft_answer", "")
        }

    # 2. Add Nodes
    workflow.add_node("understanding", node_understanding)
    workflow.add_node("retrieval", node_retrieval)
    workflow.add_node("synthesis", node_synthesis)
    workflow.add_node("validation", node_validation)
    workflow.add_node("retry_handler", node_retry)
    workflow.add_node("fallback_handler", node_fallback)
    workflow.add_node("finalize", node_finalize)

    # 3. Define Entrypoint
    workflow.set_entry_point("understanding")

    # 4. Conditional Edge: Understanding -> Next Node
    def route_after_understanding(state: AgentState) -> str:
        intent = state.get("intent", "knowledge_query")
        if intent == "knowledge_query":
            return "retrieval"
        else:
            # Casual, tool, and ambiguous queries do not invoke retrieval
            return "synthesis"

    workflow.add_conditional_edges(
        "understanding",
        route_after_understanding,
        {
            "retrieval": "retrieval",
            "synthesis": "synthesis"
        }
    )

    # Edge: Retrieval -> Synthesis
    workflow.add_edge("retrieval", "synthesis")

    # Edge: Synthesis -> Validation
    workflow.add_edge("synthesis", "validation")

    # 5. Conditional Edge: Validation -> Finalize / Retry / Fallback
    def route_after_validation(state: AgentState) -> str:
        is_valid = state.get("validation_result", True)
        retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", settings.MAX_VALIDATION_RETRIES)

        if is_valid:
            return "finalize"
        elif retries < max_retries:
            return "retry_handler"
        else:
            return "fallback_handler"

    workflow.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "finalize": "finalize",
            "retry_handler": "retry_handler",
            "fallback_handler": "fallback_handler"
        }
    )

    # Edge: Retry -> Synthesis
    workflow.add_edge("retry_handler", "synthesis")

    # Terminal Edges
    workflow.add_edge("finalize", END)
    workflow.add_edge("fallback_handler", END)

    return workflow.compile()


class AgentOrchestrator:
    """High-level wrapper for executing the LangGraph agent state machine."""

    def __init__(self, rag_pipeline: RAGPipeline, llm_client: Optional[BaseLLMClient] = None):
        self.rag_pipeline = rag_pipeline
        self.llm_client = llm_client or rag_pipeline.llm_client
        self.graph = build_agent_graph(rag_pipeline=self.rag_pipeline, llm_client=self.llm_client)

    def run(
        self,
        query: str,
        conversation_id: str,
        messages: Optional[list] = None,
        retrieval_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> AgentState:
        """Executes the LangGraph state machine from user query to final verified answer."""
        start_time = time.perf_counter()

        initial_state: AgentState = {
            "conversation_id": conversation_id,
            "query": query,
            "language": "en",
            "intent": "knowledge_query",
            "needs_retrieval": False,
            "needs_clarification": False,
            "messages": messages or [],
            "retrieval_mode": retrieval_mode or settings.RETRIEVAL_MODE,
            "top_k": top_k or settings.RETRIEVAL_TOP_K,
            "document_id": document_id,
            "min_score": min_score,
            "retrieved_chunks": [],
            "citations": [],
            "tool_results": None,
            "draft_answer": "",
            "validation_result": True,
            "validation_reason": None,
            "retry_count": 0,
            "max_retries": settings.MAX_VALIDATION_RETRIES,
            "final_answer": "",
            "timings": {}
        }

        # Execute state transitions
        final_state = self.graph.invoke(initial_state)
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        final_state["timings"]["total_ms"] = round(total_time_ms, 2)

        return final_state

    def run_stream(
        self,
        query: str,
        conversation_id: str,
        messages: Optional[list] = None,
        retrieval_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Progressively executes the multi-agent workflow:
        1. Emits 'meta' event with intent, sources, tool results.
        2. Streams text 'chunk' events in real time.
        3. Validates and emits 'done' event with final answer and latency metrics.
        """
        start_time = time.perf_counter()

        state: AgentState = {
            "conversation_id": conversation_id,
            "query": query,
            "language": "en",
            "intent": "knowledge_query",
            "needs_retrieval": False,
            "needs_clarification": False,
            "messages": messages or [],
            "retrieval_mode": retrieval_mode or settings.RETRIEVAL_MODE,
            "top_k": top_k or settings.RETRIEVAL_TOP_K,
            "document_id": document_id,
            "min_score": min_score,
            "retrieved_chunks": [],
            "citations": [],
            "tool_results": None,
            "draft_answer": "",
            "validation_result": True,
            "validation_reason": None,
            "retry_count": 0,
            "max_retries": settings.MAX_VALIDATION_RETRIES,
            "final_answer": "",
            "timings": {}
        }

        # 1. Understanding
        state = query_understanding_node(state)
        intent = state.get("intent", "knowledge_query")
        lang = state.get("language", "en")

        # 2. Retrieval if knowledge query
        if intent == "knowledge_query":
            state = retrieval_node(state, rag_pipeline=self.rag_pipeline)

        def serialize_sources(citations):
            res = []
            for c in citations:
                if hasattr(c, "model_dump"):
                    res.append(c.model_dump())
                elif hasattr(c, "dict"):
                    res.append(c.dict())
                elif isinstance(c, dict):
                    res.append(c)
                else:
                    res.append(str(c))
            return res

        sources_data = serialize_sources(state.get("citations", []))

        # 3. Emit meta event
        yield {
            "type": "meta",
            "conversation_id": conversation_id,
            "intent": intent,
            "retrieval_mode": state.get("retrieval_mode", "hybrid"),
            "sources": sources_data,
            "tool_result": None
        }

        # 4. Stream Synthesis
        t_gen_0 = time.perf_counter()
        tool_results = None
        accumulated_chunks = []

        if intent == "tool_query":
            t_tool_0 = time.perf_counter()
            tool_results = evaluate_arithmetic(query)
            tool_ms = (time.perf_counter() - t_tool_0) * 1000.0
            state.setdefault("timings", {})["tool_ms"] = round(tool_ms, 2)
            state["tool_results"] = tool_results

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

            # Stream words
            words = draft_answer.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                accumulated_chunks.append(chunk)
                yield {"type": "chunk", "text": chunk}

        elif intent == "casual_query":
            q_lower = query.lower()
            if lang == "te" or "నమస్కారం" in q_lower or "హలో" in q_lower:
                draft_answer = "నమస్కారం! నేను మీ తెలుగు మల్టీ-ఏజెంట్ AI అసిస్టెంట్‌ని. మీకు విశ్వవిద్యాలయ నిబంధనలు లేదా విద్యా సమాచారంలో ఎలా సహాయపడగలను?"
            elif "thank" in q_lower or "ధన్యవాదాలు" in q_lower:
                draft_answer = "You're welcome! Feel free to ask if you have any more questions about academic regulations."
            else:
                draft_answer = "Hello! I am your Multi-Agent Telugu Conversational AI assistant. How can I assist you with academic regulations and document queries today?"

            words = draft_answer.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                accumulated_chunks.append(chunk)
                yield {"type": "chunk", "text": chunk}

        elif intent == "ambiguous_query":
            if lang == "te":
                draft_answer = "దయచేసి మీ ప్రశ్నను మరింత వివరంగా అడగండి. మీరు ఏ అంశం లేదా నిబంధన గురించి తెలుసుకోవాలనుకుంటున్నారు?"
            else:
                draft_answer = "Could you please clarify your question? Please specify which section or regulation you would like to know about."

            words = draft_answer.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                accumulated_chunks.append(chunk)
                yield {"type": "chunk", "text": chunk}

        else:
            chunks = state.get("retrieved_chunks", [])
            user_prompt = format_context_prompt(query=query, retrieved_chunks=chunks)

            try:
                for chunk in self.llm_client.generate_stream(prompt=user_prompt):
                    if chunk:
                        accumulated_chunks.append(chunk)
                        yield {"type": "chunk", "text": chunk}
            except Exception as gen_err:
                logger.warning(f"LLM streaming generation failed ({gen_err}); falling back.")
                if chunks:
                    first_chunk = chunks[0]
                    top_text = getattr(first_chunk, "text", "") if hasattr(first_chunk, "text") else (first_chunk.get("text", "") if isinstance(first_chunk, dict) else str(first_chunk))
                    fallback_chunk = f"పత్రాల ప్రకారం: {top_text}" if lang == "te" else f"According to the documents: {top_text}"
                elif lang == "te":
                    fallback_chunk = "మీరు అడిగిన ప్రశ్నకు సంబంధించిన సమాచారం పత్రాలలో లభించలేదు."
                else:
                    fallback_chunk = "I could not find relevant information in the uploaded documents."
                accumulated_chunks.append(fallback_chunk)
                yield {"type": "chunk", "text": fallback_chunk}

        gen_ms = (time.perf_counter() - t_gen_0) * 1000.0
        state.setdefault("timings", {})["generation_ms"] = round(gen_ms, 2)
        state["draft_answer"] = "".join(accumulated_chunks)
        state["final_answer"] = state["draft_answer"]

        # 5. Validation
        state = validation_node(state)

        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        state.setdefault("timings", {})["total_ms"] = round(total_time_ms, 2)

        # 6. Emit done event
        yield {
            "type": "done",
            "query": query,
            "conversation_id": conversation_id,
            "intent": intent,
            "retrieval_mode": state.get("retrieval_mode", "hybrid"),
            "answer": state.get("final_answer", ""),
            "sources": sources_data,
            "tool_result": state.get("tool_results"),
            "validation": {
                "valid": state.get("validation_result", True),
                "reason": state.get("validation_reason"),
                "retries_used": state.get("retry_count", 0)
            },
            "latency": state.get("timings", {})
        }
