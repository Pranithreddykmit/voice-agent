"""
In-Memory BM25 Sparse Retrieval Index.
Uses BM25Plus from rank_bm25 with multilingual tokenization for Telugu and English document chunks.
BM25Plus prevents the zero/negative IDF pathology on small or micro corpora.
"""

import logging
from typing import List, Optional, Dict, Any, Set
from rank_bm25 import BM25Plus
from app.config import settings
from app.schemas import DocumentChunk, RetrievedChunk
from app.retrieval.tokenizer import tokenize_text

logger = logging.getLogger(__name__)


class BM25Index:
    """In-memory BM25Plus sparse index for document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_tokens_set: List[Set[str]] = []
        self._bm25: Optional[BM25Plus] = None
        self._chunk_map: Dict[str, DocumentChunk] = {}

    @property
    def is_indexed(self) -> bool:
        return self._bm25 is not None and len(self.chunks) > 0

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the BM25 index and rebuild the sparse model.
        """
        if not chunks:
            return

        for chunk in chunks:
            if chunk.chunk_id not in self._chunk_map:
                tokens = tokenize_text(chunk.text)
                self.chunks.append(chunk)
                self._chunk_map[chunk.chunk_id] = chunk
                self.tokenized_corpus.append(tokens)
                self.chunk_tokens_set.append(set(tokens))

        self._bm25 = BM25Plus(self.tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
        logger.info(f"BM25 index updated with {len(self.chunks)} total chunks.")

    def clear(self) -> None:
        """Clear the in-memory index."""
        self.chunks = []
        self.tokenized_corpus = []
        self.chunk_tokens_set = []
        self._bm25 = None
        self._chunk_map = {}

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Execute BM25 sparse keyword search against indexed chunks.
        """
        if not self.is_indexed:
            logger.info("BM25 index is empty. Returning 0 results.")
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        query_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)

        # Collect candidate indices with at least one matching token and positive score
        scored_candidates = []
        for idx, score in enumerate(scores):
            # Check lexical overlap
            if query_set.intersection(self.chunk_tokens_set[idx]) and score > 0.0:
                chunk = self.chunks[idx]
                if document_id and chunk.document_id != document_id:
                    continue
                scored_candidates.append((score, chunk))

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_candidates[:top_k]

        retrieved: List[RetrievedChunk] = []
        for rank, (score, chunk) in enumerate(top_results, start=1):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=float(round(score, 4)),
                    retrieval_method="bm25",
                    rank=rank
                )
            )

        logger.info(f"BM25 search returned {len(retrieved)} results for query: '{query[:40]}...'")
        return retrieved
