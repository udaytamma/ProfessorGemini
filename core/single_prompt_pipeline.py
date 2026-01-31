"""Single Prompt pipeline for context-aware generation.

Simple pipeline: Load KB context + User prompt -> Gemini -> Output.
Used when users want a single API call with the full Knowledge Base as context.

Supports two modes:
1. RAG mode: Semantic retrieval of top-k relevant documents (~150KB)
2. Full context mode: All documents from gemini-responses/ (~2.5MB)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from config.settings import get_settings
from core.context_loader import ContextLoader
from core.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies and startup overhead
_rag_retriever = None
_sync_performed = False


@dataclass(slots=True)
class SinglePromptResult:
    """Result from Single Prompt execution.

    Attributes:
        session_id: Unique identifier for this execution.
        prompt: The user's original prompt.
        output: Generated markdown content.
        context_file_count: Number of KB documents loaded.
        context_chars: Total characters in context.
        duration_ms: Total execution time in milliseconds.
        success: Whether generation succeeded.
        error: Error message if failed.
        rag_used: Whether RAG retrieval was used (vs full context).

    Note: Uses slots=True for ~20% memory reduction per instance.
    """

    session_id: str
    prompt: str
    output: str
    context_file_count: int
    context_chars: int
    duration_ms: int
    success: bool
    error: Optional[str] = None
    rag_used: bool = False


def _get_rag_retriever():
    """Lazy initialization of RAG retriever with sync."""
    global _rag_retriever, _sync_performed

    settings = get_settings()

    # Only initialize if RAG is available
    if not settings.is_rag_available():
        return None

    # Perform sync once on first use
    if not _sync_performed:
        try:
            from core.document_syncer import sync_if_needed

            logger.info("Checking if document sync is needed...")
            result = sync_if_needed()
            if result:
                for source, stats in result.items():
                    logger.info(
                        "Synced %s: %d indexed, %d skipped, %d deleted",
                        source, stats.indexed, stats.skipped, stats.deleted
                    )
            _sync_performed = True
        except Exception as e:
            logger.warning("Document sync failed: %s", e)
            _sync_performed = True  # Don't retry on failure

    # Create retriever if not exists
    if _rag_retriever is None:
        try:
            from core.rag_retriever import RAGRetriever

            _rag_retriever = RAGRetriever()
        except Exception as e:
            logger.warning("Failed to initialize RAGRetriever: %s", e)
            return None

    return _rag_retriever


class SinglePromptPipeline:
    """Executes single prompt with Knowledge Base context.

    This pipeline:
    1. Loads context via RAG (top-k relevant docs) or full context
    2. Sends a single Gemini API call with context + user prompt
    3. Returns the generated output

    RAG mode reduces context from ~2.5MB to ~150KB (94% reduction).
    Falls back to full context if RAG fails.
    """

    def __init__(self) -> None:
        """Initialize pipeline with dependencies."""
        self._settings = get_settings()
        self._gemini = GeminiClient()
        self._context_loader = ContextLoader(
            self._settings.get_gemini_responses_path()
        )

    def execute(self, prompt: str) -> SinglePromptResult:
        """Execute single prompt with KB context.

        Uses RAG for semantic retrieval if available, falls back to full context.

        Args:
            prompt: User's prompt/request.

        Returns:
            SinglePromptResult with generated content or error.
        """
        session_id = str(uuid4())[:8]
        start_time = time.time()
        rag_used = False

        logger.info("[%s] Starting Single Prompt execution", session_id)

        # Step 1: Load context (RAG or full)
        rag_retriever = _get_rag_retriever()

        if rag_retriever is not None:
            # Try RAG retrieval
            logger.info("[%s] Using RAG retrieval (top-%d)...", session_id, self._settings.rag_top_k)
            context_result = rag_retriever.get_context_for_prompt(prompt)

            if context_result.success:
                rag_used = True
                logger.info(
                    "[%s] RAG retrieved %d docs (%d chars)",
                    session_id, context_result.file_count, context_result.total_chars
                )
            else:
                # Fallback to full context
                logger.warning(
                    "[%s] RAG failed (%s), falling back to full context",
                    session_id, context_result.error
                )
                context_result = self._context_loader.load_all_documents()
        else:
            # RAG not available, use full context
            logger.info("[%s] Loading full Knowledge Base context...", session_id)
            context_result = self._context_loader.load_all_documents()

        if not context_result.success:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error("[%s] Failed to load context: %s", session_id, context_result.error)
            return SinglePromptResult(
                session_id=session_id,
                prompt=prompt,
                output="",
                context_file_count=0,
                context_chars=0,
                duration_ms=duration_ms,
                success=False,
                error=f"Failed to load Knowledge Base: {context_result.error}",
                rag_used=rag_used,
            )

        logger.info(
            "[%s] Context ready: %d documents (%d chars) [RAG=%s]",
            session_id, context_result.file_count, context_result.total_chars, rag_used
        )

        # Step 2: Generate with Gemini
        logger.info("[%s] Calling Gemini with context...", session_id)
        response = self._gemini.generate_with_context(
            prompt=prompt,
            context=context_result.content,
            file_count=context_result.file_count,
            char_count=context_result.total_chars,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        if not response.success:
            logger.error("[%s] Gemini call failed: %s", session_id, response.error)
            return SinglePromptResult(
                session_id=session_id,
                prompt=prompt,
                output="",
                context_file_count=context_result.file_count,
                context_chars=context_result.total_chars,
                duration_ms=duration_ms,
                success=False,
                error=f"Gemini generation failed: {response.error}",
                rag_used=rag_used,
            )

        # Calculate timing breakdown
        rag_ms = context_result.rag_duration_ms or 0
        gemini_ms = response.duration_ms
        other_ms = duration_ms - rag_ms - gemini_ms

        logger.info(
            "[%s] Single Prompt completed in %dms [RAG=%dms, Gemini=%dms, Other=%dms] [RAG_enabled=%s]",
            session_id, duration_ms, rag_ms, gemini_ms, other_ms, rag_used
        )

        return SinglePromptResult(
            session_id=session_id,
            prompt=prompt,
            output=response.content,
            context_file_count=context_result.file_count,
            context_chars=context_result.total_chars,
            duration_ms=duration_ms,
            success=True,
            rag_used=rag_used,
        )
