"""
Cross-Encoder Reranker Subsystem.
Reranks candidate document chunks against the user query using joint cross-attention.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from app.config import settings
from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for candidate reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 4
    ) -> List[RetrievedChunk]:
        """Rerank candidates and return the top-K highest-scoring chunks."""
        pass


class TransformersCrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder Reranker using HuggingFace Transformers (e.g. BAAI/bge-reranker-base).
    Computes joint cross-attention logits for (query, passage) pairs.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._tokenizer = None
        self._model = None

    def _lazy_load(self):
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                logger.info(f"Loading Cross-Encoder model '{self.model_name}'...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.eval()
                logger.info(f"Cross-Encoder model '{self.model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Transformers cross-encoder '{self.model_name}': {str(e)}")
                raise RuntimeError(f"Could not load cross-encoder '{self.model_name}': {str(e)}") from e

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 4
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates[:top_k]

        try:
            self._lazy_load()
            import torch

            pairs = [[query, chunk.text] for chunk in candidates]
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                scores = self._model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()

            # Pair candidates with their new cross-encoder scores
            scored_candidates = list(zip(scores, candidates))
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            reranked_results: List[RetrievedChunk] = []
            for rank, (score, base_chunk) in enumerate(scored_candidates[:top_k], start=1):
                reranked_chunk = RetrievedChunk(
                    chunk_id=base_chunk.chunk_id,
                    document_id=base_chunk.document_id,
                    source=base_chunk.source,
                    page=base_chunk.page,
                    chunk_index=base_chunk.chunk_index,
                    text=base_chunk.text,
                    score=float(round(score, 4)),
                    retrieval_method="rerank",
                    rank=rank
                )
                reranked_results.append(reranked_chunk)

            logger.info(f"Cross-encoder reranked {len(candidates)} candidates down to {len(reranked_results)}.")
            return reranked_results

        except Exception as e:
            logger.warning(f"Cross-encoder inference failed ({str(e)}). Falling back to original candidate ordering.")
            # Fallback: maintain candidate ordering but slice to top_k
            return candidates[:top_k]


class MockReranker(BaseReranker):
    """
    Deterministic Mock Reranker for tests and environments without PyTorch/Transformers.
    Computes a deterministic semantic keyword overlap score.
    """

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 4
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        query_words = set(query.lower().split())

        def score_candidate(chunk: RetrievedChunk) -> float:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            return chunk.score + (overlap * 0.1)

        scored = [(score_candidate(c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrievedChunk] = []
        for rank, (score, c) in enumerate(scored[:top_k], start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    source=c.source,
                    page=c.page,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    score=float(round(score, 4)),
                    retrieval_method="rerank(mock)",
                    rank=rank
                )
            )
        return results


def get_reranker() -> BaseReranker:
    """Factory function returning the configured reranker."""
    provider = settings.RERANKER_PROVIDER.lower()
    if provider in ["transformers", "local"]:
        return TransformersCrossEncoderReranker(model_name=settings.RERANKER_MODEL)
    elif provider == "mock":
        return MockReranker()
    else:
        logger.warning(f"Unknown RERANKER_PROVIDER '{provider}'. Falling back to MockReranker.")
        return MockReranker()
