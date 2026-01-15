"""Bar Raiser adversarial critique loop.

Implements the core adversarial loop where Gemini drafts content
and Claude reviews it as a Mag7 Bar Raiser.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from config.settings import CriticStrictness, get_settings
from core.gemini_client import GeminiClient, GeminiResponse
from core.claude_client import ClaudeClient, CritiqueResult, ClaudeResponse


logger = logging.getLogger(__name__)


@dataclass
class AttemptRecord:
    """Record of a single draft-critique attempt.

    Attributes:
        attempt_number: Which attempt this was (1-based).
        strictness: Strictness level used for critique.
        draft: The draft content.
        critique_passed: Whether critique passed.
        critique_feedback: Feedback from the critique.
        draft_duration_ms: Time for draft generation.
        critique_duration_ms: Time for critique.
    """

    attempt_number: int
    strictness: CriticStrictness
    draft: str
    critique_passed: bool
    critique_feedback: str
    draft_duration_ms: int
    critique_duration_ms: int


@dataclass
class BarRaiserResult:
    """Final result from the Bar Raiser loop.

    Attributes:
        topic: The topic that was processed.
        final_content: The final approved (or best) content.
        low_confidence: True if content never passed review.
        attempts: List of all attempt records.
        total_duration_ms: Total time spent on this topic.
        success: Whether processing completed (even if low confidence).
        error: Error message if processing failed entirely.
    """

    topic: str
    final_content: str
    low_confidence: bool
    attempts: list[AttemptRecord] = field(default_factory=list)
    total_duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None


class BarRaiser:
    """Orchestrates the adversarial draft-critique loop.

    For each topic:
    1. Gemini drafts the section
    2. Claude critiques as Bar Raiser
    3. If FAIL, Gemini rewrites with feedback
    4. Repeat up to max_retries times
    5. On 3rd attempt, strictness relaxes to MEDIUM
    6. If still fails, accept with low_confidence flag
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        claude_client: ClaudeClient,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize Bar Raiser.

        Args:
            gemini_client: Client for Gemini API.
            claude_client: Client for Claude API.
            status_callback: Optional callback for status updates.
        """
        self._gemini = gemini_client
        self._claude = claude_client
        self._settings = get_settings()
        self._status_callback = status_callback or (lambda x: None)

    def _log_status(self, message: str) -> None:
        """Log status and call callback.

        Args:
            message: Status message.
        """
        logger.info(message)
        self._status_callback(message)

    def process_topic(
        self,
        topic: str,
        context: str,
        worker_id: int,
    ) -> BarRaiserResult:
        """Process a single topic through the Bar Raiser loop.

        Args:
            topic: The topic to write about.
            context: Context from base knowledge.
            worker_id: Worker ID for logging.

        Returns:
            BarRaiserResult with final content and metadata.
        """
        attempts: list[AttemptRecord] = []
        total_duration = 0
        best_draft = ""
        passed = False

        max_attempts = self._settings.max_retries + 1  # +1 for initial attempt
        accumulated_feedback = ""

        for attempt_num in range(1, max_attempts + 1):
            # Determine strictness level
            # Attempts 1-2: HIGH, Attempt 3+: MEDIUM
            if attempt_num <= 2:
                strictness = CriticStrictness.HIGH
            else:
                strictness = CriticStrictness.MEDIUM

            self._log_status(
                f"Worker {worker_id}: Attempt {attempt_num}/{max_attempts} "
                f"for '{topic[:40]}...' (strictness: {strictness.value})"
            )

            # Generate draft
            if attempt_num == 1:
                # First attempt - fresh draft
                draft_response = self._gemini.generate_section_draft(
                    topic=topic,
                    context=context,
                    feedback=accumulated_feedback,
                )
            else:
                # Rewrite with critique feedback
                draft_response = self._gemini.rewrite_section(
                    topic=topic,
                    previous_draft=best_draft,
                    critique=accumulated_feedback,
                    strictness=strictness.value,
                )

            total_duration += draft_response.duration_ms

            if not draft_response.success:
                self._log_status(
                    f"Worker {worker_id}: Draft failed - {draft_response.error}"
                )
                # Continue to next attempt if possible
                continue

            best_draft = draft_response.content

            # Critique the draft
            self._log_status(f"Worker {worker_id}: Critiquing draft...")

            critique_result, critique_response = self._claude.critique_draft(
                draft=best_draft,
                topic=topic,
                strictness=strictness,
            )

            total_duration += critique_response.duration_ms

            # Record this attempt
            attempt_record = AttemptRecord(
                attempt_number=attempt_num,
                strictness=strictness,
                draft=best_draft,
                critique_passed=critique_result.passed,
                critique_feedback=critique_result.feedback,
                draft_duration_ms=draft_response.duration_ms,
                critique_duration_ms=critique_response.duration_ms,
            )
            attempts.append(attempt_record)

            if critique_result.passed:
                self._log_status(f"Worker {worker_id}: APPROVED on attempt {attempt_num}")
                passed = True
                break
            else:
                self._log_status(
                    f"Worker {worker_id}: REJECTED - {critique_result.feedback[:100]}..."
                )
                accumulated_feedback = critique_result.feedback

        # Determine final status
        if passed:
            self._log_status(f"Worker {worker_id}: Completed '{topic[:40]}...' - APPROVED")
        else:
            self._log_status(
                f"Worker {worker_id}: Completed '{topic[:40]}...' - LOW CONFIDENCE"
            )

        return BarRaiserResult(
            topic=topic,
            final_content=best_draft,
            low_confidence=not passed,
            attempts=attempts,
            total_duration_ms=total_duration,
            success=True,
        )

    def process_topics_parallel(
        self,
        topics: list[str],
        context: str,
        executor,
    ) -> list[BarRaiserResult]:
        """Process multiple topics in parallel using provided executor.

        Args:
            topics: List of topics to process.
            context: Shared context from base knowledge.
            executor: ThreadPoolExecutor to use for parallelism.

        Returns:
            List of BarRaiserResult, one per topic.
        """
        futures = []

        for i, topic in enumerate(topics):
            worker_id = i + 1
            future = executor.submit(
                self.process_topic,
                topic=topic,
                context=context,
                worker_id=worker_id,
            )
            futures.append((topic, future))

        # Collect results
        results = []
        for topic, future in futures:
            try:
                result = future.result(timeout=self._settings.api_timeout * 4)
                results.append(result)
            except TimeoutError:
                logger.error(f"Timeout processing topic: {topic}")
                results.append(
                    BarRaiserResult(
                        topic=topic,
                        final_content="",
                        low_confidence=True,
                        success=False,
                        error="Processing timeout",
                    )
                )
            except Exception as e:
                logger.error(f"Error processing topic '{topic}': {e}")
                results.append(
                    BarRaiserResult(
                        topic=topic,
                        final_content="",
                        low_confidence=True,
                        success=False,
                        error=str(e),
                    )
                )

        return results
