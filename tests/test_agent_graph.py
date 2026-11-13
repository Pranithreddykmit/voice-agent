"""
Integration tests for LangGraph Multi-Agent State Machine.
"""

import uuid
import pytest
from app.rag import RAGPipeline
from app.storage.qdrant import QdrantVectorStore
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import MockReranker
from app.generation.llm import MockLLMClient
from app.agents.graph import AgentOrchestrator, build_agent_graph


@pytest.fixture
def agent_orchestrator(mock_embedding_provider, mock_llm_client):
    test_coll = f"test_agent_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(collection_name=test_coll)
    bm25 = BM25Index()
    reranker = MockReranker()

    pipeline = RAGPipeline(
        embedding_provider=mock_embedding_provider,
        vector_store=store,
        bm25_index=bm25,
        reranker=reranker,
        llm_client=mock_llm_client
    )
    orchestrator = AgentOrchestrator(rag_pipeline=pipeline, llm_client=mock_llm_client)
    yield orchestrator

    try:
        if store.is_connected():
            store.client.delete_collection(test_coll)
    except Exception:
        pass


def test_agent_graph_knowledge_path(agent_orchestrator, sample_pdf_bytes):
    if not agent_orchestrator.rag_pipeline.vector_store.is_connected():
        pytest.skip("Qdrant vector database container is offline; run 'docker compose up -d' for live integration tests.")
    
    # Ingest document
    agent_orchestrator.rag_pipeline.ingest_pdf(sample_pdf_bytes, filename="rules.pdf")


    state = agent_orchestrator.run(
        query="What is the minimum attendance requirement?",
        conversation_id="conv_kb_01"
    )

    assert state["intent"] == "knowledge_query"
    assert state["needs_retrieval"] is True
    assert len(state["retrieved_chunks"]) >= 1
    assert len(state["citations"]) >= 1
    assert state["validation_result"] is True
    assert len(state["final_answer"]) > 0
    assert state["timings"]["total_ms"] > 0


def test_agent_graph_tool_path(agent_orchestrator):
    state = agent_orchestrator.run(
        query="What is 15% of 240?",
        conversation_id="conv_tool_01"
    )

    assert state["intent"] == "tool_query"
    assert state["needs_retrieval"] is False
    assert len(state["retrieved_chunks"]) == 0
    assert state["tool_results"] is not None
    assert state["tool_results"]["result"] == 36
    assert "36" in state["final_answer"]
    assert state["validation_result"] is True


def test_agent_graph_casual_path(agent_orchestrator):
    state = agent_orchestrator.run(
        query="Hello! How are you today?",
        conversation_id="conv_casual_01"
    )

    assert state["intent"] == "casual_query"
    assert state["needs_retrieval"] is False
    assert len(state["retrieved_chunks"]) == 0
    assert state["validation_result"] is True
    assert "AI assistant" in state["final_answer"] or "Hello" in state["final_answer"]


def test_agent_graph_ambiguous_path(agent_orchestrator):
    state = agent_orchestrator.run(
        query="What about the second one?",
        conversation_id="conv_ambig_01",
        messages=[]
    )

    assert state["intent"] == "ambiguous_query"
    assert state["needs_clarification"] is True
    assert "clarify" in state["final_answer"].lower() or "వివరంగా" in state["final_answer"]


def test_agent_graph_bounded_retry_and_fallback(agent_orchestrator, sample_pdf_bytes):
    if not agent_orchestrator.rag_pipeline.vector_store.is_connected():
        pytest.skip("Qdrant vector database container is offline; run 'docker compose up -d' for live integration tests.")

    # Ingest document
    agent_orchestrator.rag_pipeline.ingest_pdf(sample_pdf_bytes, filename="rules.pdf")

    # Create an invalid response generator
    invalid_llm = MockLLMClient(fixed_response="Hallucinated claim from Page 99 with no valid source.")
    orchestrator = AgentOrchestrator(rag_pipeline=agent_orchestrator.rag_pipeline, llm_client=invalid_llm)

    state = orchestrator.run(
        query="What is the attendance rule?",
        conversation_id="conv_retry_01"
    )

    # Should hit retry limit and execute fallback
    assert state["retry_count"] >= 1
    assert "apologize" in state["final_answer"].lower() or "క్షమించండి" in state["final_answer"]
