"""
Qdrant Vector Database Integration.
Manages collection creation, payload indexing, point upsert, and cosine similarity search.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.schemas import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class QdrantStorageError(Exception):
    """Raised when communication or operations with Qdrant fail."""
    pass


class QdrantVectorStore:
    """Wrapper around Qdrant client for document chunk vector storage and retrieval."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            logger.info(f"Initializing Qdrant client for URL: {self.url}")
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=10.0,
                check_compatibility=False
            )
        return self._client

    def is_connected(self) -> bool:
        """Check if Qdrant cluster/server is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant healthcheck failed: {str(e)}")
            return False

    def ensure_collection(self, vector_dimension: int) -> None:
        """
        Ensure that the target collection exists with the specified vector dimension and Cosine metric.
        If dimension mismatch occurs with an existing collection, re-creates it.
        """
        try:
            if self.client.collection_exists(self.collection_name):
                collection_info = self.client.get_collection(self.collection_name)
                # Check vector size compatibility
                existing_size = getattr(collection_info.config.params.vectors, "size", None)
                if existing_size is not None and existing_size != vector_dimension:
                    logger.warning(
                        f"Collection '{self.collection_name}' has dimension {existing_size}, "
                        f"but active model produces dimension {vector_dimension}. Recreating collection..."
                    )
                    self.client.delete_collection(self.collection_name)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=rest_models.VectorParams(
                            size=vector_dimension,
                            distance=rest_models.Distance.COSINE
                        )
                    )
                    logger.info(f"Collection '{self.collection_name}' re-created with dimension {vector_dimension}.")
            else:
                logger.info(
                    f"Collection '{self.collection_name}' does not exist. "
                    f"Creating with vector size {vector_dimension} and Cosine distance."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=vector_dimension,
                        distance=rest_models.Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' successfully created.")
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection '{self.collection_name}': {str(e)}", exc_info=True)
            raise QdrantStorageError(f"Qdrant collection initialization error: {str(e)}") from e

    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str) -> str:
        """Convert any string chunk_id deterministically into a valid UUID string for Qdrant."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        vectors: List[List[float]]
    ) -> None:
        """
        Batch upsert document chunks and their dense embeddings into Qdrant.
        """
        if not chunks or not vectors:
            return

        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch between chunk count ({len(chunks)}) and vector count ({len(vectors)})")

        self.ensure_collection(vector_dimension=len(vectors[0]))

        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = self._chunk_id_to_uuid(chunk.chunk_id)
            payload: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "char_count": chunk.char_count
            }
            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        try:
            # Batch upsert in groups of 100 points
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            logger.info(f"Successfully indexed {len(points)} chunks into Qdrant collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to upsert chunks into Qdrant: {str(e)}", exc_info=True)
            raise QdrantStorageError(f"Qdrant indexing error: {str(e)}") from e

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """
        Execute dense vector cosine similarity search in Qdrant using query_points.
        Returns empty list if collection does not exist or has no matching points.
        """
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' does not exist yet. Returning empty search results.")
                return []

            query_filter = None
            if document_id:
                query_filter = rest_models.Filter(
                    must=[
                        rest_models.FieldCondition(
                            key="document_id",
                            match=rest_models.MatchValue(value=document_id)
                        )
                    ]
                )

            # Use client.query_points for Qdrant client v1.10+
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=min_score,
                with_payload=True
            )

            retrieved: List[RetrievedChunk] = []
            for hit in query_response.points:
                payload = hit.payload or {}
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=str(payload.get("chunk_id", hit.id)),
                        document_id=str(payload.get("document_id", "")),
                        source=str(payload.get("source", "unknown")),
                        page=int(payload.get("page", 1)),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        text=str(payload.get("text", "")),
                        score=float(hit.score)
                    )
                )

            logger.info(f"Qdrant query returned {len(retrieved)} results for top_k={top_k}.")
            return retrieved

        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.info(f"Collection '{self.collection_name}' not found during query. Returning empty.")
                return []
            logger.error(f"Qdrant search error: {str(e)}", exc_info=True)
            raise QdrantStorageError(f"Qdrant search failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Qdrant search error: {str(e)}", exc_info=True)
            raise QdrantStorageError(f"Qdrant search failed: {str(e)}") from e
