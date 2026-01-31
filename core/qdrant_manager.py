"""Qdrant abstraction layer for Professor Gemini.

Provides unified interface for all Qdrant operations:
- Embedding generation
- Document upsert/delete
- Semantic search
- Collection management

Based on patterns from IngredientScanner/tools/ingredient_lookup.py
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config.settings import get_settings

logger = logging.getLogger(__name__)

# LRU cache for search results
_search_cache: OrderedDict[str, tuple[list, float]] = OrderedDict()
_cache_lock = Lock()
_CACHE_MAX_SIZE = 100
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _cache_key(query: str, top_k: int, source_filter: Optional[str]) -> str:
    """Generate cache key from search parameters."""
    return f"{query}|{top_k}|{source_filter or ''}"


def _get_cached_result(key: str) -> Optional[list]:
    """Get cached result if valid, None otherwise."""
    with _cache_lock:
        if key in _search_cache:
            results, timestamp = _search_cache[key]
            if time.time() - timestamp < _CACHE_TTL_SECONDS:
                # Move to end (most recently used)
                _search_cache.move_to_end(key)
                logger.debug("Cache hit for query: %s", key[:50])
                return results
            else:
                # Expired, remove it
                del _search_cache[key]
    return None


def _set_cached_result(key: str, results: list) -> None:
    """Cache search results with timestamp."""
    with _cache_lock:
        _search_cache[key] = (results, time.time())
        # Evict oldest if over size limit
        while len(_search_cache) > _CACHE_MAX_SIZE:
            _search_cache.popitem(last=False)

VECTOR_SIZE = 768  # gemini-embedding-001 with output_dimensionality=768
EMBEDDING_MODEL = "gemini-embedding-001"


@dataclass(slots=True)
class QdrantDocument:
    """Document stored in Qdrant.

    Attributes:
        doc_id: Unique identifier (format: source:slug).
        source: Document source (kb, scratch, interview).
        title: Document title extracted from content.
        content: Full document content.
        content_hash: MD5 hash for change detection.
        indexed_at: ISO timestamp of indexing.
        char_count: Content length in characters.
        metadata: Source-specific metadata.
        score: Similarity score (populated on search).

    Note: Uses slots=True for ~20% memory reduction per instance.
    """

    doc_id: str
    source: str
    title: str
    content: str
    content_hash: str
    indexed_at: str
    char_count: int
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class QdrantManager:
    """Manages all Qdrant operations for Professor Gemini.

    Provides a clean interface for:
    - Document indexing (upsert)
    - Document deletion
    - Semantic search
    - Collection management
    """

    def __init__(self) -> None:
        """Initialize manager with lazy client creation."""
        self._settings = get_settings()
        self._client: QdrantClient | None = None
        self._genai: genai.Client | None = None

    @property
    def collection_name(self) -> str:
        """Get the collection name from settings."""
        return self._settings.rag_collection

    def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client.

        Returns:
            Configured QdrantClient instance.

        Raises:
            ValueError: If Qdrant is not configured.
        """
        if self._client is None:
            if not self._settings.is_qdrant_configured():
                raise ValueError(
                    "Qdrant not configured. Check QDRANT_URL and QDRANT_API_KEY."
                )
            self._client = QdrantClient(
                url=self._settings.qdrant_url,
                api_key=self._settings.qdrant_api_key,
            )
        return self._client

    def _get_genai(self) -> genai.Client:
        """Get or create GenAI client for embeddings.

        Returns:
            Configured genai.Client instance.

        Raises:
            ValueError: If Gemini API is not configured.
        """
        if self._genai is None:
            if not self._settings.is_gemini_configured():
                raise ValueError("Gemini API not configured. Check GEMINI_API_KEY.")
            self._genai = genai.Client(api_key=self._settings.gemini_api_key)
        return self._genai

    def ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        client = self._get_client()
        collections = client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if not exists:
            logger.info("Creating collection: %s", self.collection_name)
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def get_embedding(
        self, text: str, task_type: str = "RETRIEVAL_QUERY"
    ) -> list[float]:
        """Get embedding vector for text.

        Args:
            text: Text to embed.
            task_type: "RETRIEVAL_QUERY" for search, "RETRIEVAL_DOCUMENT" for indexing.

        Returns:
            768-dimensional embedding vector.
        """
        genai_client = self._get_genai()
        result = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=VECTOR_SIZE,
            ),
        )
        return result.embeddings[0].values

    def upsert(self, doc: QdrantDocument) -> bool:
        """Add or update a document in Qdrant.

        Args:
            doc: Document to upsert.

        Returns:
            True if successful.
        """
        client = self._get_client()
        self.ensure_collection_exists()

        # Embed using RETRIEVAL_DOCUMENT for indexing
        # Use title + first 2000 chars for embedding (captures key content)
        embed_text = f"{doc.title}\n\n{doc.content[:2000]}"
        embedding = self.get_embedding(embed_text, task_type="RETRIEVAL_DOCUMENT")

        point = PointStruct(
            id=hash(doc.doc_id) % (2**63),  # Stable int ID from doc_id
            vector=embedding,
            payload={
                "doc_id": doc.doc_id,
                "source": doc.source,
                "title": doc.title,
                "content": doc.content,
                "content_hash": doc.content_hash,
                "indexed_at": doc.indexed_at,
                "char_count": doc.char_count,
                "metadata": doc.metadata,
            },
        )

        client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        return True

    def delete(self, doc_id: str) -> bool:
        """Delete a document by doc_id.

        Args:
            doc_id: Document ID to delete (e.g., "kb:error-budgets").

        Returns:
            True if successful.
        """
        client = self._get_client()
        point_id = hash(doc_id) % (2**63)

        client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )
        logger.info("Deleted document: %s", doc_id)
        return True

    def search(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
        use_cache: bool = True,
    ) -> list[QdrantDocument]:
        """Semantic search for documents.

        Args:
            query: Search query text.
            top_k: Number of results to return (defaults to settings.rag_top_k).
            source_filter: Optional filter by source (e.g., "kb").
            use_cache: Whether to use LRU cache (default True).

        Returns:
            List of matching documents with scores.
        """
        if top_k is None:
            top_k = self._settings.rag_top_k

        # Check cache first (if enabled)
        cache_key = _cache_key(query, top_k, source_filter)
        if use_cache:
            cached = _get_cached_result(cache_key)
            if cached is not None:
                logger.info("RAG cache hit for query: %s...", query[:30])
                return cached

        client = self._get_client()
        self.ensure_collection_exists()

        # Time embedding generation
        embed_start = time.time()
        embedding = self.get_embedding(query, task_type="RETRIEVAL_QUERY")
        embed_ms = int((time.time() - embed_start) * 1000)

        # Build filter if source specified
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        # Time Qdrant query
        query_start = time.time()
        results = client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        query_ms = int((time.time() - query_start) * 1000)

        logger.info("RAG timing: embedding=%dms, qdrant_query=%dms", embed_ms, query_ms)

        documents = []
        for point in results.points:
            payload = point.payload or {}
            documents.append(
                QdrantDocument(
                    doc_id=payload.get("doc_id", ""),
                    source=payload.get("source", ""),
                    title=payload.get("title", ""),
                    content=payload.get("content", ""),
                    content_hash=payload.get("content_hash", ""),
                    indexed_at=payload.get("indexed_at", ""),
                    char_count=payload.get("char_count", 0),
                    metadata=payload.get("metadata", {}),
                    score=point.score,
                )
            )

        # Cache the results
        if use_cache:
            _set_cached_result(cache_key, documents)

        return documents

    def list_documents(self, source_filter: str | None = None) -> list[dict]:
        """List all indexed documents (without full content).

        Args:
            source_filter: Optional filter by source.

        Returns:
            List of document summaries.
        """
        client = self._get_client()
        self.ensure_collection_exists()

        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        results = client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=1000,
            with_payload=[
                "doc_id",
                "source",
                "title",
                "content_hash",
                "indexed_at",
                "char_count",
            ],
        )

        return [
            {
                "doc_id": p.payload.get("doc_id"),
                "source": p.payload.get("source"),
                "title": p.payload.get("title"),
                "content_hash": p.payload.get("content_hash"),
                "indexed_at": p.payload.get("indexed_at"),
                "char_count": p.payload.get("char_count"),
            }
            for p in results[0]
        ]

    def get_document(self, doc_id: str) -> QdrantDocument | None:
        """Get a single document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Document if found, None otherwise.
        """
        client = self._get_client()
        point_id = hash(doc_id) % (2**63)

        try:
            results = client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
            )
            if results:
                payload = results[0].payload or {}
                return QdrantDocument(
                    doc_id=payload.get("doc_id", ""),
                    source=payload.get("source", ""),
                    title=payload.get("title", ""),
                    content=payload.get("content", ""),
                    content_hash=payload.get("content_hash", ""),
                    indexed_at=payload.get("indexed_at", ""),
                    char_count=payload.get("char_count", 0),
                    metadata=payload.get("metadata", {}),
                )
        except Exception as e:
            logger.warning("Error retrieving document %s: %s", doc_id, e)
        return None

    def get_collection_stats(self) -> dict:
        """Get collection statistics.

        Returns:
            Dict with points_count, status, etc.
        """
        client = self._get_client()
        try:
            info = client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "status": info.status.value if hasattr(info.status, "value") else str(info.status),
            }
        except Exception as e:
            logger.warning("Error getting collection stats: %s", e)
            return {"points_count": 0, "status": "not_found"}
