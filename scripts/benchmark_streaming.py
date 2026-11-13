"""
Benchmark script to measure lightweight streaming performance:
- Time To First Token (TTFT)
- Total stream duration
- Number of chunks
- Final response correctness
"""

import time
import json
from fastapi.testclient import TestClient

from app.main import app, get_pipeline, get_orchestrator
from app.embeddings.provider import BaseEmbeddingProvider
from app.generation.llm import MockLLMClient
from app.storage.qdrant import QdrantVectorStore
from app.rag import RAGPipeline
from app.agents.graph import AgentOrchestrator


class MockEmbeddingProvider(BaseEmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 64
    def embed_documents(self, texts):
        return [[0.1] * 64 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 64


def measure_streaming_performance():
    # Setup lightweight mock pipeline
    pipeline = RAGPipeline(
        embedding_provider=MockEmbeddingProvider(),
        vector_store=QdrantVectorStore(collection_name="benchmark_coll"),
        llm_client=MockLLMClient()
    )
    orchestrator = AgentOrchestrator(rag_pipeline=pipeline)

    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator

    with TestClient(app) as client:
        # Benchmark 1: Casual Query
        print("\n--- Benchmark 1: Casual Query ---")
        t_start = time.perf_counter()
        t_first_chunk = None
        chunks = []

        with client.stream("POST", "/query/stream", json={"query": "హలో, నమస్కారం"}) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data.get("type") == "chunk":
                    if t_first_chunk is None:
                        t_first_chunk = time.perf_counter()
                    chunks.append(data.get("text", ""))

        t_end = time.perf_counter()
        ttft_ms = (t_first_chunk - t_start) * 1000.0 if t_first_chunk else 0.0
        total_duration_ms = (t_end - t_start) * 1000.0
        full_text = "".join(chunks)

        print(f"Time to First Chunk (TTFT): {ttft_ms:.2f} ms")
        print(f"Total Stream Duration:     {total_duration_ms:.2f} ms")
        print(f"Total Chunks Streamed:     {len(chunks)}")
        print(f"Final Response:            {full_text}")

        # Benchmark 2: Knowledge Query
        print("\n--- Benchmark 2: Knowledge Query ---")
        t_start = time.perf_counter()
        t_first_chunk = None
        chunks = []

        with client.stream("POST", "/query/stream", json={"query": "What is the attendance policy?"}) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data.get("type") == "chunk":
                    if t_first_chunk is None:
                        t_first_chunk = time.perf_counter()
                    chunks.append(data.get("text", ""))

        t_end = time.perf_counter()
        ttft_ms = (t_first_chunk - t_start) * 1000.0 if t_first_chunk else 0.0
        total_duration_ms = (t_end - t_start) * 1000.0
        full_text = "".join(chunks)

        print(f"Time to First Chunk (TTFT): {ttft_ms:.2f} ms")
        print(f"Total Stream Duration:     {total_duration_ms:.2f} ms")
        print(f"Total Chunks Streamed:     {len(chunks)}")
        print(f"Final Response:            {full_text}")

    app.dependency_overrides.clear()


if __name__ == "__main__":
    measure_streaming_performance()
