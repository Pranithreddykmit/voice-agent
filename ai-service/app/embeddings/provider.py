"""
Configurable Embedding Provider.
Supports local high-performance multilingual models (e.g. intfloat/multilingual-e5-large via FastEmbed ONNX)
and cloud embedding APIs (OpenAI text-embedding-3-small) behind a unified interface.
Properly handles model-specific prefixing (e.g. 'passage: ' and 'query: ' for E5 models).
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a list of text passages."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate dense vector embedding for a single search query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension of the embedding model."""
        pass


class FastEmbedProvider(BaseEmbeddingProvider):
    """
    Local ONNX-based embedding provider using FastEmbed.
    Default model: 'intfloat/multilingual-e5-large' (1024-dim, high-quality Telugu & English coverage).
    Automatically applies 'passage: ' and 'query: ' prefixes required by the E5 model family.
    """

    def __init__(self, model_name: Optional[str] = None, batch_size: int = 32):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size
        self._model = None
        self._is_e5 = "e5" in self.model_name.lower()
        # Default dimension for multilingual-e5-large is 1024
        self._dimension = 1024 if "large" in self.model_name else settings.EMBEDDING_DIMENSION

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            logger.info(f"Initializing FastEmbed TextEmbedding model '{self.model_name}'...")
            self._model = TextEmbedding(model_name=self.model_name)
            logger.info(f"FastEmbed model '{self.model_name}' loaded successfully.")
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def _prepare_document_text(self, text: str) -> str:
        """Apply passage prefix if required by E5 architecture."""
        if self._is_e5 and not text.startswith("passage: "):
            return f"passage: {text}"
        return text

    def _prepare_query_text(self, text: str) -> str:
        """Apply query prefix if required by E5 architecture."""
        if self._is_e5 and not text.startswith("query: "):
            return f"query: {text}"
        return text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            formatted_texts = [self._prepare_document_text(t) for t in texts]
            embeddings_gen = self.model.embed(formatted_texts, batch_size=self.batch_size)
            vectors = [vec.tolist() if isinstance(vec, np.ndarray) else list(vec) for vec in embeddings_gen]
            if vectors and len(vectors[0]) != self._dimension:
                self._dimension = len(vectors[0])
            return vectors
        except Exception as e:
            logger.error(f"FastEmbed document embedding error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Query text cannot be empty.")
        formatted_query = self._prepare_query_text(text)
        try:
            embeddings_gen = self.model.embed([formatted_query], batch_size=1)
            vectors = [vec.tolist() if isinstance(vec, np.ndarray) else list(vec) for vec in embeddings_gen]
            if not vectors:
                raise RuntimeError("FastEmbed returned no embedding for query.")
            return vectors[0]
        except Exception as e:
            logger.error(f"FastEmbed query embedding error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate query embedding: {str(e)}") from e


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI API-based embedding provider (e.g. text-embedding-3-small).
    """

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAIEmbeddingProvider.")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self._dimension = 1536 if "small" in model_name or "ada" in model_name else 3072

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            batch_size = 100
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(input=batch, model=self.model_name)
                for item in response.data:
                    all_embeddings.append(item.embedding)
            return all_embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}", exc_info=True)
            raise RuntimeError(f"OpenAI embedding failed: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Query text cannot be empty.")
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding


_embedding_instance: Optional[BaseEmbeddingProvider] = None


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Factory function returning the configured embedding provider singleton."""
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance

    provider_type = settings.EMBEDDING_PROVIDER.lower()
    if provider_type == "fastembed":
        _embedding_instance = FastEmbedProvider(
            model_name=settings.EMBEDDING_MODEL,
            batch_size=settings.EMBEDDING_BATCH_SIZE
        )
    elif provider_type == "openai":
        _embedding_instance = OpenAIEmbeddingProvider(
            model_name=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
    else:
        logger.warning(f"Unknown EMBEDDING_PROVIDER '{provider_type}', falling back to fastembed.")
        _embedding_instance = FastEmbedProvider(model_name="intfloat/multilingual-e5-large")

    return _embedding_instance

