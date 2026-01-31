"""RAG retriever for semantic search.

Uses QdrantManager to search for relevant documents
and formats context for Gemini. Returns the same interface
as ContextLoader for seamless integration.
"""

import logging
import re
import time

from config.settings import get_settings
from core.context_loader import LoadedContext
from core.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant documents using semantic search.

    Provides the same interface as ContextLoader but returns
    only the top-k most relevant documents instead of all.
    """

    # Pattern to match YAML frontmatter
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

    def __init__(self) -> None:
        """Initialize retriever with dependencies."""
        self._settings = get_settings()
        self._qdrant = QdrantManager()

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content.

        Args:
            content: Raw markdown content with potential frontmatter.

        Returns:
            Content with frontmatter removed.
        """
        return self.FRONTMATTER_PATTERN.sub("", content).strip()

    def get_context_for_prompt(self, query: str) -> LoadedContext:
        """Get formatted context for Gemini prompt.

        Performs semantic search and formats top-k documents
        as context string.

        Args:
            query: User's query/prompt.

        Returns:
            LoadedContext with relevant documents concatenated.
        """
        try:
            rag_start = time.time()
            top_k = self._settings.rag_top_k
            documents = self._qdrant.search(query, top_k=top_k)
            rag_duration_ms = int((time.time() - rag_start) * 1000)

            if not documents:
                logger.warning("No documents found for query")
                return LoadedContext(
                    content="",
                    file_count=0,
                    total_chars=0,
                    success=False,
                    error="No relevant documents found in knowledge base",
                )

            # Format documents as context (matching ContextLoader format)
            context_parts = []
            for doc in documents:
                # Strip frontmatter from content
                clean_content = self._strip_frontmatter(doc.content)
                # Format like ContextLoader: --- Document: filename ---
                doc_section = f"--- Document: {doc.doc_id}.md ---\n{clean_content}"
                context_parts.append(doc_section)

            content = "\n\n".join(context_parts)

            # Log retrieval stats with timing (lazy formatting)
            if logger.isEnabledFor(logging.INFO):
                scores = [f"{d.score:.3f}" for d in documents]
                logger.info(
                    "RAG retrieved %d docs in %dms (scores: %s, %d chars)",
                    len(documents), rag_duration_ms, ", ".join(scores), len(content)
                )

            return LoadedContext(
                content=content,
                file_count=len(documents),
                total_chars=len(content),
                success=True,
                rag_duration_ms=rag_duration_ms,
            )

        except Exception as e:
            logger.error("RAG retrieval failed: %s", e)
            return LoadedContext(
                content="",
                file_count=0,
                total_chars=0,
                success=False,
                error=str(e),
            )

    def search_documents(
        self, query: str, top_k: int | None = None
    ) -> list[dict]:
        """Search for documents and return metadata.

        Useful for debugging or showing which documents were retrieved.

        Args:
            query: Search query.
            top_k: Number of results (defaults to settings).

        Returns:
            List of document metadata with scores.
        """
        try:
            documents = self._qdrant.search(query, top_k=top_k)
            return [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "score": doc.score,
                    "char_count": doc.char_count,
                }
                for doc in documents
            ]
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []
