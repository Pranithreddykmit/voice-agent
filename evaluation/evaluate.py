"""
Retrieval Evaluation and Benchmark Harness.
Compares Dense, BM25, Hybrid (RRF), and Hybrid + Cross-Encoder Reranking across English, Telugu, and mixed queries.
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

from app.config import settings
from app.rag import RAGPipeline
from app.schemas import QueryRequest
from evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    ndcg_at_k
)


def run_retrieval_benchmark(
    dataset_path: str = "evaluation/dataset.json",
    pdf_path: str = "sample_data/academic_regulations.pdf",
    output_json_path: str = "evaluation/benchmark_results.json"
):
    print("=" * 88)
    print("PHASE 2: ADVANCED RAG RETRIEVAL BENCHMARK EVALUATION")
    print("=" * 88)

    # 1. Load Evaluation Dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Filter positive queries (in-corpus) for retrieval metrics
    in_corpus_queries = [q for q in dataset if q.get("is_in_corpus", True)]
    negative_queries = [q for q in dataset if not q.get("is_in_corpus", True)]

    print(f"Loaded {len(dataset)} total evaluation queries:")
    print(f"  - In-corpus test queries: {len(in_corpus_queries)} (English, Telugu, Code-mixed)")
    print(f"  - Out-of-corpus negative queries: {len(negative_queries)}")
    print("-" * 88)

    # 2. Initialize RAG Pipeline and Index Document
    pipeline = RAGPipeline()
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        print(f"Ingesting corpus '{pdf_path}' into Qdrant and BM25...")
        upload_res = pipeline.ingest_pdf(pdf_bytes, filename=os.path.basename(pdf_path))
        print(f"Ingested {upload_res.total_pages} pages into {upload_res.total_chunks} chunks ({upload_res.processing_time_ms}ms).")
    else:
        print(f"Warning: Corpus PDF '{pdf_path}' not found. Ensure document is indexed.")

    modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]
    benchmark_results: Dict[str, Any] = {}

    print("\nExecuting evaluation experiments across all retrieval modes...\n")

    for mode in modes:
        mode_latencies: List[float] = []
        hits_1: List[float] = []
        hits_3: List[float] = []
        hits_5: List[float] = []
        recalls_3: List[float] = []
        recalls_5: List[float] = []
        mrrs_5: List[float] = []
        ndcgs_5: List[float] = []

        for item in in_corpus_queries:
            query = item["question"]
            ground_truth_pages = item["relevant_pages"]

            t0 = time.perf_counter()
            retrieved_chunks, timings = pipeline.retrieve(
                query=query,
                mode=mode,
                top_k=5
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            mode_latencies.append(lat_ms)

            retrieved_pages = [c.page for c in retrieved_chunks]

            hits_1.append(hit_at_k(retrieved_pages, ground_truth_pages, k=1))
            hits_3.append(hit_at_k(retrieved_pages, ground_truth_pages, k=3))
            hits_5.append(hit_at_k(retrieved_pages, ground_truth_pages, k=5))
            recalls_3.append(recall_at_k(retrieved_pages, ground_truth_pages, k=3))
            recalls_5.append(recall_at_k(retrieved_pages, ground_truth_pages, k=5))
            mrrs_5.append(reciprocal_rank(retrieved_pages, ground_truth_pages, k=5))
            ndcgs_5.append(ndcg_at_k(retrieved_pages, ground_truth_pages, k=5))

        benchmark_results[mode] = {
            "mode": mode,
            "queries_evaluated": len(in_corpus_queries),
            "hit@1": round(sum(hits_1) / len(hits_1), 4),
            "hit@3": round(sum(hits_3) / len(hits_3), 4),
            "hit@5": round(sum(hits_5) / len(hits_5), 4),
            "recall@3": round(sum(recalls_3) / len(recalls_3), 4),
            "recall@5": round(sum(recalls_5) / len(recalls_5), 4),
            "mrr@5": round(sum(mrrs_5) / len(mrrs_5), 4),
            "ndcg@5": round(sum(ndcgs_5) / len(ndcgs_5), 4),
            "avg_latency_ms": round(sum(mode_latencies) / len(mode_latencies), 2)
        }

    # 3. Format and Display Comparison Results
    print("=" * 88)
    print(f"{'Retrieval Mode':<20} | {'Hit@1':<8} | {'Hit@3':<8} | {'Recall@5':<9} | {'MRR@5':<8} | {'nDCG@5':<8} | {'Avg Latency':<12}")
    print("-" * 88)
    for mode in modes:
        res = benchmark_results[mode]
        print(
            f"{res['mode']:<20} | "
            f"{res['hit@1']:<8.4f} | "
            f"{res['hit@3']:<8.4f} | "
            f"{res['recall@5']:<9.4f} | "
            f"{res['mrr@5']:<8.4f} | "
            f"{res['ndcg@5']:<8.4f} | "
            f"{res['avg_latency_ms']} ms"
        )
    print("=" * 88)

    # 4. Out-of-Context Refusal Evaluation
    print("\n--- Negative Out-of-Context Queries Verification ---")
    neg_results = []
    for neg in negative_queries:
        req = QueryRequest(query=neg["question"], min_score=0.75)
        resp = pipeline.query(req)
        is_refusal = "does not contain" in resp.answer.lower() or "సమాచారం లేదు" in resp.answer
        neg_results.append({
            "id": neg["id"],
            "query": neg["question"],
            "refused_correctly": is_refusal,
            "sources_count": len(resp.sources)
        })
        print(f"[{neg['id']}] Refusal Handled: {is_refusal} (Sources: {len(resp.sources)})")

    # Save to JSON
    output_payload = {
        "benchmark_summary": benchmark_results,
        "negative_controls": neg_results,
        "dataset_size": len(dataset),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved benchmark results to '{output_json_path}'.")

    return benchmark_results


if __name__ == "__main__":
    run_retrieval_benchmark()
