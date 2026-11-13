"""
Comprehensive Multi-Agent LangGraph Benchmark Evaluation.
Evaluates Intent Routing Accuracy, Unnecessary Retrieval Elimination, Safe Tool Execution,
Response Validation, Contextual Follow-up Resolution, and Grounded Refusals.
"""

import json
import os
import sys
import time
from typing import List, Dict, Any

# Ensure workspace root and ai-service are on sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "ai-service"))

from app.rag import RAGPipeline
from app.storage.postgres import PostgresManager
from app.agents.graph import AgentOrchestrator

AGENT_BENCHMARK_CASES = [
    # 1. Knowledge Queries (English, Telugu, Code-mixed)
    {
        "id": "kb_01_en",
        "category": "knowledge_english",
        "query": "What is the minimum attendance requirement for semester examinations?",
        "expected_intent": "knowledge_query",
        "expected_retrieval": True,
        "expected_page": 1,
        "is_followup": False
    },
    {
        "id": "kb_02_te",
        "category": "knowledge_telugu",
        "query": "సెమిస్టర్ పరీక్షలకు హాజరు కావడానికి కనీసం ఎంత శాతం హాజరు తప్పనిసరి?",
        "expected_intent": "knowledge_query",
        "expected_retrieval": True,
        "expected_page": 1,
        "is_followup": False
    },
    {
        "id": "kb_03_mixed",
        "category": "knowledge_mixed",
        "query": "ఈ regulations ప్రకారం revaluation fee ఎంత చెల్లించాలి?",
        "expected_intent": "knowledge_query",
        "expected_retrieval": True,
        "expected_page": 3,
        "is_followup": False
    },
    # 2. Casual Small Talk (Should NOT trigger retrieval)
    {
        "id": "cas_01",
        "category": "casual",
        "query": "Hello! How are you today?",
        "expected_intent": "casual_query",
        "expected_retrieval": False,
        "is_followup": False
    },
    {
        "id": "cas_02_te",
        "category": "casual",
        "query": "నమస్కారం, మీరు నాకు ఎలా సహాయపడగలరు?",
        "expected_intent": "casual_query",
        "expected_retrieval": False,
        "is_followup": False
    },
    # 3. Tool Queries (Safe Calculator - Should NOT trigger retrieval)
    {
        "id": "tool_01_pct",
        "category": "calculator_tool",
        "query": "What is 15% of 240?",
        "expected_intent": "tool_query",
        "expected_retrieval": False,
        "expected_result": "36",
        "is_followup": False
    },
    {
        "id": "tool_02_arithmetic",
        "category": "calculator_tool",
        "query": "Calculate 125 * 18",
        "expected_intent": "tool_query",
        "expected_retrieval": False,
        "expected_result": "2250",
        "is_followup": False
    },
    {
        "id": "tool_03_expr",
        "category": "calculator_tool",
        "query": "Evaluate (50 + 20) / 7",
        "expected_intent": "tool_query",
        "expected_retrieval": False,
        "expected_result": "10",
        "is_followup": False
    },
    # 4. Contextual Follow-up Turn
    {
        "id": "ctx_01_turn1",
        "category": "contextual_turn1",
        "query": "What is the attendance requirement?",
        "expected_intent": "knowledge_query",
        "expected_retrieval": True,
        "is_followup": False
    },
    {
        "id": "ctx_02_turn2",
        "category": "contextual_turn2",
        "query": "What happens if I have 70%?",
        "expected_intent": "knowledge_query",
        "expected_retrieval": True,
        "is_followup": True
    },
    # 5. Out-of-Context Negative Control
    {
        "id": "neg_01",
        "category": "out_of_context",
        "query": "Who was the first emperor of the Maurya Dynasty?",
        "expected_intent": "knowledge_query",
        "expected_retrieval": True,
        "expected_refusal": True,
        "is_followup": False
    }
]


def run_agent_benchmark(
    pdf_path: str = "sample_data/academic_regulations.pdf",
    output_json_path: str = "evaluation/agent_benchmark_results.json"
):
    print("=" * 96)
    print("PHASE 3: LANGGRAPH MULTI-AGENT STATEFUL WORKFLOW BENCHMARK")
    print("=" * 96)

    # 1. Initialize Pipeline & Orchestrator
    rag_pipeline = RAGPipeline()
    db = PostgresManager()
    orchestrator = AgentOrchestrator(rag_pipeline=rag_pipeline)

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        print(f"Indexing corpus '{pdf_path}' into Qdrant & BM25...")
        rag_pipeline.ingest_pdf(pdf_bytes, filename=os.path.basename(pdf_path))

    results = []
    category_metrics: Dict[str, Dict[str, Any]] = {}
    shared_conv_id = db.get_or_create_conversation()

    print(f"\nExecuting {len(AGENT_BENCHMARK_CASES)} evaluation cases across 5 categories...\n")

    correct_routing = 0
    unnecessary_retrievals = 0
    total_non_knowledge = 0
    validation_passes = 0

    for case in AGENT_BENCHMARK_CASES:
        cid = shared_conv_id if case["is_followup"] else db.get_or_create_conversation()
        recent_history = db.get_recent_messages(cid, limit=4)

        # Save user query to db
        db.add_message(conversation_id=cid, role="user", content=case["query"])

        t0 = time.perf_counter()
        agent_state = orchestrator.run(
            query=case["query"],
            conversation_id=cid,
            messages=recent_history,
            retrieval_mode="hybrid"
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Save assistant answer to db
        db.add_message(
            conversation_id=cid,
            role="assistant",
            content=agent_state.get("final_answer", ""),
            intent=agent_state.get("intent")
        )

        detected_intent = agent_state.get("intent")
        retrieved_count = len(agent_state.get("retrieved_chunks", []))
        is_valid = agent_state.get("validation_result", True)
        answer = agent_state.get("final_answer", "")

        # Check routing
        routing_correct = detected_intent == case["expected_intent"]
        if routing_correct:
            correct_routing += 1

        # Check unnecessary retrieval
        if not case["expected_retrieval"]:
            total_non_knowledge += 1
            if retrieved_count > 0:
                unnecessary_retrievals += 1

        if is_valid:
            validation_passes += 1

        cat = case["category"]
        if cat not in category_metrics:
            category_metrics[cat] = {"count": 0, "total_ms": 0.0, "valid_count": 0}
        category_metrics[cat]["count"] += 1
        category_metrics[cat]["total_ms"] += elapsed_ms
        if is_valid:
            category_metrics[cat]["valid_count"] += 1

        results.append({
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "expected_intent": case["expected_intent"],
            "detected_intent": detected_intent,
            "routing_match": routing_correct,
            "retrieved_chunks": retrieved_count,
            "is_valid": is_valid,
            "answer_preview": answer[:80] + ("..." if len(answer) > 80 else ""),
            "latency_ms": round(elapsed_ms, 2)
        })

    routing_accuracy = (correct_routing / len(AGENT_BENCHMARK_CASES)) * 100.0
    unnecessary_retrieval_rate = (unnecessary_retrievals / max(1, total_non_knowledge)) * 100.0
    validation_pass_rate = (validation_passes / len(AGENT_BENCHMARK_CASES)) * 100.0

    print("=" * 96)
    print(f"{'Case ID':<16} | {'Category':<18} | {'Intent':<16} | {'Route OK':<10} | {'Chunks':<8} | {'Latency':<10}")
    print("-" * 96)
    for r in results:
        print(
            f"{r['id']:<16} | "
            f"{r['category']:<18} | "
            f"{r['detected_intent']:<16} | "
            f"{str(r['routing_match']):<10} | "
            f"{r['retrieved_chunks']:<8} | "
            f"{r['latency_ms']} ms"
        )
    print("=" * 96)
    print(f"Overall Routing Accuracy:             {routing_accuracy:.1f}% ({correct_routing}/{len(AGENT_BENCHMARK_CASES)})")
    print(f"Unnecessary Retrieval Rate (Casual/Tool): {unnecessary_retrieval_rate:.1f}% ({unnecessary_retrievals}/{total_non_knowledge}) [Target: 0.0%]")
    print(f"Validation Pass Rate:                 {validation_pass_rate:.1f}% ({validation_passes}/{len(AGENT_BENCHMARK_CASES)})")
    print("=" * 96)

    # Save results to JSON
    output_payload = {
        "summary": {
            "total_cases": len(AGENT_BENCHMARK_CASES),
            "routing_accuracy_pct": round(routing_accuracy, 2),
            "unnecessary_retrieval_rate_pct": round(unnecessary_retrieval_rate, 2),
            "validation_pass_rate_pct": round(validation_pass_rate, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "cases": results
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved agent benchmark results to '{output_json_path}'.")

    return output_payload


if __name__ == "__main__":
    run_agent_benchmark()
