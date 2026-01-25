"""Single Prompt pipeline for context-aware generation.

Simple pipeline: Load KB context + User prompt -> Gemini -> Output.
Used when users want a single API call with the full Knowledge Base as context.
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


@dataclass
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
    """

    session_id: str
    prompt: str
    output: str
    context_file_count: int
    context_chars: int
    duration_ms: int
    success: bool
    error: Optional[str] = None


class SinglePromptPipeline:
    """Executes single prompt with Knowledge Base context.

    This pipeline:
    1. Loads all markdown files from gemini-responses/ as context
    2. Sends a single Gemini API call with context + user prompt
    3. Returns the generated output

    Unlike the Deep Dive pipeline (4 steps, parallel calls), this is
    a simple single-call workflow for freeform generation.
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

        Args:
            prompt: User's prompt/request.

        Returns:
            SinglePromptResult with generated content or error.
        """
        session_id = str(uuid4())[:8]
        start_time = time.time()

        logger.info(f"[{session_id}] Starting Single Prompt execution")

        # Step 1: Load Knowledge Base context
        logger.info(f"[{session_id}] Loading Knowledge Base context...")
        context_result = self._context_loader.load_all_documents()

        if not context_result.success:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[{session_id}] Failed to load context: {context_result.error}")
            return SinglePromptResult(
                session_id=session_id,
                prompt=prompt,
                output="",
                context_file_count=0,
                context_chars=0,
                duration_ms=duration_ms,
                success=False,
                error=f"Failed to load Knowledge Base: {context_result.error}",
            )

        logger.info(
            f"[{session_id}] Loaded {context_result.file_count} documents "
            f"({context_result.total_chars:,} chars)"
        )

        # Step 2: Generate with Gemini
        logger.info(f"[{session_id}] Calling Gemini with context...")
        response = self._gemini.generate_with_context(
            prompt=prompt,
            context=context_result.content,
            file_count=context_result.file_count,
            char_count=context_result.total_chars,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        if not response.success:
            logger.error(f"[{session_id}] Gemini call failed: {response.error}")
            return SinglePromptResult(
                session_id=session_id,
                prompt=prompt,
                output="",
                context_file_count=context_result.file_count,
                context_chars=context_result.total_chars,
                duration_ms=duration_ms,
                success=False,
                error=f"Gemini generation failed: {response.error}",
            )

        logger.info(
            f"[{session_id}] Single Prompt completed in {duration_ms}ms "
            f"(Gemini: {response.duration_ms}ms)"
        )

        return SinglePromptResult(
            session_id=session_id,
            prompt=prompt,
            output=response.content,
            context_file_count=context_result.file_count,
            context_chars=context_result.total_chars,
            duration_ms=duration_ms,
            success=True,
        )
