"""
Core RAG Pipeline Coordinator.
Orchestrates document parsing, structure chunking, multilingual embeddings,
Qdrant dense indexing, BM25 sparse indexing, Reciprocal Rank Fusion (RRF),
Cross-Encoder Reranking, grounded LLM generation, and granular latency profiling.
"""

import hashlib
import logging
import time
from typing import Optional, List, Dict, Any, Tuple

from app.config import settings
from app.schemas import (
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    SourceCitation,
    LatencyMetrics,
    RetrievedChunk,
    ValidationInfo
)
from app.ingestion.pdf_loader import PDFLoader, PDFExtractionError
from app.ingestion.chunker import StructureAwareChunker
from app.embeddings.provider import BaseEmbeddingProvider, get_embedding_provider
from app.storage.qdrant import QdrantVectorStore, QdrantStorageError
from app.retrieval.bm25 import BM25Index
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.reranker import BaseReranker, get_reranker
from app.retrieval.query_expansion import expand_retrieval_query
from app.generation.llm import BaseLLMClient, get_llm_client, LLMGenerationError
from app.generation.prompts import format_context_prompt

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end multi-mode RAG workflow coordinator."""

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[BaseReranker] = None,
        llm_client: Optional[BaseLLMClient] = None,
        chunker: Optional[StructureAwareChunker] = None
    ):
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = vector_store or QdrantVectorStore()
        self.bm25_index = bm25_index or BM25Index(k1=settings.BM25_K1, b=settings.BM25_B)
        self.reranker = reranker or get_reranker()
        self.llm_client = llm_client or get_llm_client()
        self.chunker = chunker or StructureAwareChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def ingest_pdf(
        self,
        file_bytes: bytes,
        filename: str
    ) -> DocumentUploadResponse:
        """
        Ingests a PDF: extracts pages, chunks text, generates dense embeddings,
        indexes into Qdrant, and updates the in-memory BM25 sparse index.
        """
        start_time = time.perf_counter()

        if not file_bytes:
            raise PDFExtractionError(f"Uploaded file '{filename}' is empty.")

        content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        doc_id = f"doc_{content_hash}"

        # 1. Parse pages
        pages = PDFLoader.load_pdf(file_input=file_bytes, source_name=filename)
        total_pages = len(pages)

        # 2. Chunk text
        chunks = self.chunker.chunk_pages(
            pages=pages,
            document_id=doc_id,
            source_name=filename
        )

        if not chunks:
            logger.warning(f"No extractable chunks found in '{filename}'.")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return DocumentUploadResponse(
                document_id=doc_id,
                filename=filename,
                total_pages=total_pages,
                total_chunks=0,
                collection_name=self.vector_store.collection_name,
                status="warning_no_text",
                processing_time_ms=round(elapsed_ms, 2)
            )

        # 3. Dense Indexing (Qdrant)
        chunk_texts = [c.text for c in chunks]
        vectors = self.embedding_provider.embed_documents(chunk_texts)
        self.vector_store.upsert_chunks(chunks=chunks, vectors=vectors)

        # 4. Sparse Indexing (BM25)
        self.bm25_index.add_chunks(chunks)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"Successfully indexed '{filename}' ({total_pages} pages, {len(chunks)} chunks) "
            f"into Qdrant & BM25 in {elapsed_ms:.2f}ms."
        )

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=filename,
            total_pages=total_pages,
            total_chunks=len(chunks),
            collection_name=self.vector_store.collection_name,
            status="success",
            processing_time_ms=round(elapsed_ms, 2)
        )

    def retrieve(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """
        Executes multi-mode candidate retrieval and ranking.
        Supports: 'dense', 'bm25', 'hybrid', 'hybrid_rerank'.
        Returns (retrieved_chunks, latency_timings_dict).
        """
        retrieval_mode = (mode or settings.RETRIEVAL_MODE).lower()
        k = top_k or settings.RETRIEVAL_TOP_K
        pool_size = max(k * 2, settings.RERANK_TOP_N)

        # Non-destructive query expansion for internal dense & sparse candidate search
        search_query = expand_retrieval_query(query)

        timings: Dict[str, float] = {
            "dense_ms": 0.0,
            "bm25_ms": 0.0,
            "rrf_ms": 0.0,
            "rerank_ms": 0.0,
            "total_retrieval_ms": 0.0
        }
        retrieval_start = time.perf_counter()

        if retrieval_mode == "dense":
            t0 = time.perf_counter()
            query_vector = self.embedding_provider.embed_query(search_query)
            chunks = self.vector_store.search(
                query_vector=query_vector,
                top_k=k,
                document_id=document_id,
                min_score=min_score
            )
            for idx, c in enumerate(chunks, start=1):
                c.retrieval_method = "dense"
                c.rank = idx
            timings["dense_ms"] = (time.perf_counter() - t0) * 1000.0
            final_chunks = chunks

        elif retrieval_mode == "bm25":
            t0 = time.perf_counter()
            chunks = self.bm25_index.search(
                query=search_query,
                top_k=k,
                document_id=document_id
            )
            timings["bm25_ms"] = (time.perf_counter() - t0) * 1000.0
            final_chunks = chunks

        elif retrieval_mode == "hybrid":
            t_dense = time.perf_counter()
            query_vector = self.embedding_provider.embed_query(search_query)
            dense_candidates = self.vector_store.search(
                query_vector=query_vector,
                top_k=pool_size,
                document_id=document_id,
                min_score=min_score
            )
            timings["dense_ms"] = (time.perf_counter() - t_dense) * 1000.0

            t_bm25 = time.perf_counter()
            bm25_candidates = self.bm25_index.search(
                query=search_query,
                top_k=pool_size,
                document_id=document_id
            )
            timings["bm25_ms"] = (time.perf_counter() - t_bm25) * 1000.0

            if not dense_candidates and not bm25_candidates:
                final_chunks = []
            else:
                t_rrf = time.perf_counter()
                fused_candidates = reciprocal_rank_fusion(
                    ranked_lists={"dense": dense_candidates, "bm25": bm25_candidates},
                    k=settings.RRF_K,
                    top_k=k
                )
                timings["rrf_ms"] = (time.perf_counter() - t_rrf) * 1000.0
                final_chunks = fused_candidates

        elif retrieval_mode == "hybrid_rerank":
            t_dense = time.perf_counter()
            query_vector = self.embedding_provider.embed_query(search_query)
            dense_candidates = self.vector_store.search(
                query_vector=query_vector,
                top_k=pool_size,
                document_id=document_id,
                min_score=min_score
            )
            timings["dense_ms"] = (time.perf_counter() - t_dense) * 1000.0

            t_bm25 = time.perf_counter()
            bm25_candidates = self.bm25_index.search(
                query=search_query,
                top_k=pool_size,
                document_id=document_id
            )
            timings["bm25_ms"] = (time.perf_counter() - t_bm25) * 1000.0

            if not dense_candidates and not bm25_candidates:
                final_chunks = []
            else:
                t_rrf = time.perf_counter()
                fused_candidates = reciprocal_rank_fusion(
                    ranked_lists={"dense": dense_candidates, "bm25": bm25_candidates},
                    k=settings.RRF_K,
                    top_k=pool_size
                )
                timings["rrf_ms"] = (time.perf_counter() - t_rrf) * 1000.0

                t_rerank = time.perf_counter()
                reranked_chunks = self.reranker.rerank(
                    query=search_query,
                    candidates=fused_candidates,
                    top_k=k
                )
                timings["rerank_ms"] = (time.perf_counter() - t_rerank) * 1000.0
                final_chunks = reranked_chunks

        else:
            logger.warning(f"Unknown retrieval mode '{retrieval_mode}'. Falling back to dense retrieval.")
            return self.retrieve(query=query, mode="dense", top_k=k, document_id=document_id, min_score=min_score)

        timings["total_retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000.0
        return final_chunks, timings

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Executes direct query retrieval (via selected mode) and grounded LLM generation with latency breakdown.
        """
        total_start = time.perf_counter()
        active_mode = request.retrieval_mode or settings.RETRIEVAL_MODE

        # 1. Retrieval
        retrieved_chunks, timings = self.retrieve(
            query=request.query,
            mode=active_mode,
            top_k=request.top_k,
            document_id=request.document_id,
            min_score=request.min_score
        )

        # 2. Generation
        generation_start = time.perf_counter()
        user_prompt = format_context_prompt(query=request.query, retrieved_chunks=retrieved_chunks)
        answer = self.llm_client.generate(prompt=user_prompt)
        generation_time_ms = (time.perf_counter() - generation_start) * 1000.0

        total_time_ms = (time.perf_counter() - total_start) * 1000.0

        # 3. Source Citations
        sources: List[SourceCitation] = [
            SourceCitation(
                document=chunk.source,
                page=chunk.page,
                score=round(chunk.score, 4),
                chunk_id=chunk.chunk_id,
                snippet=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                retrieval_method=chunk.retrieval_method,
                rank=chunk.rank
            )
            for chunk in retrieved_chunks
        ]

        return QueryResponse(
            query=request.query,
            conversation_id=request.conversation_id or "conv_direct_rag",
            intent="knowledge_query",
            retrieval_mode=active_mode,
            answer=answer,
            sources=sources,
            validation=ValidationInfo(
                valid=True,
                reason="Direct RAG baseline query executed."
            ),
            latency=LatencyMetrics(
                dense_ms=round(timings.get("dense_ms", 0.0), 2),
                bm25_ms=round(timings.get("bm25_ms", 0.0), 2),
                rrf_ms=round(timings.get("rrf_ms", 0.0), 2),
                rerank_ms=round(timings.get("rerank_ms", 0.0), 2),
                retrieval_ms=round(timings.get("total_retrieval_ms", 0.0), 2),
                generation_ms=round(generation_time_ms, 2),
                total_ms=round(total_time_ms, 2)
            )
        )
