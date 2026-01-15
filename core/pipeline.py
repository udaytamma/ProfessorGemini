"""Main pipeline orchestration for Professor Gemini.

Coordinates the 4-step hybrid AI pipeline:
1. Base Knowledge (Gemini)
2. Topic Split (Claude)
3. Deep Dive with Bar Raiser (Gemini + Claude, parallel)
4. Synthesis (Claude)
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
from uuid import uuid4

from config.settings import get_settings
from core.gemini_client import GeminiClient
from core.claude_client import ClaudeClient
from core.bar_raiser import BarRaiser, BarRaiserResult


logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """Record of a single pipeline step.

    Attributes:
        name: Step name.
        started_at: When step started.
        completed_at: When step completed.
        duration_ms: Duration in milliseconds.
        success: Whether step succeeded.
        error: Error message if failed.
        metadata: Additional step-specific data.
    """

    name: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Final result from pipeline execution.

    Attributes:
        session_id: Unique session identifier.
        topic: Original user topic.
        master_guide: Final synthesized guide content.
        low_confidence_sections: Number of sections with low confidence.
        total_sections: Total number of sections processed.
        steps: List of step records.
        deep_dive_results: Results from each topic deep dive.
        total_duration_ms: Total pipeline duration.
        success: Whether pipeline completed successfully.
        error: Error message if pipeline failed.
    """

    session_id: str
    topic: str
    master_guide: str
    low_confidence_sections: int
    total_sections: int
    steps: list[PipelineStep] = field(default_factory=list)
    deep_dive_results: list[BarRaiserResult] = field(default_factory=list)
    total_duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None


class Pipeline:
    """Orchestrates the full Professor Gemini pipeline.

    Manages the 4-step process from topic input to Master Guide output,
    with parallel processing for deep dives and comprehensive logging.
    """

    def __init__(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize pipeline with API clients.

        Args:
            status_callback: Optional callback for status updates.
        """
        self._settings = get_settings()
        self._gemini = GeminiClient()
        self._claude = ClaudeClient()
        self._status_callback = status_callback or (lambda x: None)

    def _log_status(self, message: str) -> None:
        """Log status and call callback.

        Args:
            message: Status message.
        """
        logger.info(message)
        self._status_callback(message)

    def is_ready(self) -> tuple[bool, str]:
        """Check if pipeline is ready to execute.

        Returns:
            Tuple of (ready, message).
        """
        issues = []

        if not self._gemini.is_available():
            issues.append("Gemini API not configured")

        if not self._claude.is_available():
            issues.append("Claude API not configured")

        if issues:
            return False, "; ".join(issues)

        return True, "Pipeline ready"

    def execute(self, topic: str) -> PipelineResult:
        """Execute the full pipeline for a given topic.

        Args:
            topic: The topic to deep dive into.

        Returns:
            PipelineResult with master guide and metadata.
        """
        session_id = str(uuid4())
        steps: list[PipelineStep] = []
        start_time = datetime.now()

        self._log_status(f"Starting pipeline for: {topic[:50]}...")
        self._log_status(f"Session ID: {session_id}")

        # Check readiness
        ready, message = self.is_ready()
        if not ready:
            return PipelineResult(
                session_id=session_id,
                topic=topic,
                master_guide="",
                low_confidence_sections=0,
                total_sections=0,
                success=False,
                error=message,
            )

        # Step 1: Base Knowledge
        self._log_status("Step 1: Generating base knowledge with Gemini...")
        step1 = PipelineStep(name="base_knowledge", started_at=datetime.now())

        base_response = self._gemini.generate_base_knowledge(topic)

        step1.completed_at = datetime.now()
        step1.duration_ms = base_response.duration_ms
        step1.success = base_response.success
        step1.metadata = {
            "model": base_response.model,
            "content_length": len(base_response.content),
        }

        if not base_response.success:
            step1.error = base_response.error
            steps.append(step1)
            return PipelineResult(
                session_id=session_id,
                topic=topic,
                master_guide="",
                low_confidence_sections=0,
                total_sections=0,
                steps=steps,
                success=False,
                error=f"Step 1 failed: {base_response.error}",
            )

        steps.append(step1)
        self._log_status(f"Step 1 complete: {len(base_response.content)} chars generated")

        # Step 2: Split into topics
        self._log_status("Step 2: Splitting into sub-topics with Claude...")
        step2 = PipelineStep(name="split_topics", started_at=datetime.now())

        sub_topics, split_response = self._claude.split_into_topics(base_response.content)

        step2.completed_at = datetime.now()
        step2.duration_ms = split_response.duration_ms
        step2.success = split_response.success and len(sub_topics) > 0
        step2.metadata = {
            "model": split_response.model,
            "topic_count": len(sub_topics),
            "topics": sub_topics,
        }

        if not step2.success:
            step2.error = split_response.error or "No topics extracted"
            steps.append(step2)
            return PipelineResult(
                session_id=session_id,
                topic=topic,
                master_guide="",
                low_confidence_sections=0,
                total_sections=0,
                steps=steps,
                success=False,
                error=f"Step 2 failed: {step2.error}",
            )

        steps.append(step2)
        self._log_status(f"Step 2 complete: Split into {len(sub_topics)} sub-topics")
        for i, t in enumerate(sub_topics, 1):
            self._log_status(f"  {i}. {t[:60]}...")

        # Step 3: Deep dive with Bar Raiser (parallel)
        self._log_status(
            f"Step 3: Deep dive with Bar Raiser ({self._settings.max_workers} workers)..."
        )
        step3 = PipelineStep(name="deep_dive", started_at=datetime.now())

        bar_raiser = BarRaiser(
            gemini_client=self._gemini,
            claude_client=self._claude,
            status_callback=self._status_callback,
        )

        with ThreadPoolExecutor(max_workers=self._settings.max_workers) as executor:
            deep_dive_results = bar_raiser.process_topics_parallel(
                topics=sub_topics,
                context=base_response.content,
                executor=executor,
            )

        step3.completed_at = datetime.now()
        step3.duration_ms = sum(r.total_duration_ms for r in deep_dive_results)

        # Count results
        successful = [r for r in deep_dive_results if r.success]
        low_confidence = [r for r in deep_dive_results if r.low_confidence]

        step3.success = len(successful) > 0
        step3.metadata = {
            "total_topics": len(sub_topics),
            "successful": len(successful),
            "low_confidence": len(low_confidence),
            "failed": len(deep_dive_results) - len(successful),
        }

        steps.append(step3)
        self._log_status(
            f"Step 3 complete: {len(successful)}/{len(sub_topics)} topics processed, "
            f"{len(low_confidence)} low confidence"
        )

        # Step 4: Synthesis
        self._log_status("Step 4: Synthesizing Master Guide with Claude...")
        step4 = PipelineStep(name="synthesis", started_at=datetime.now())

        # Prepare sections for synthesis
        sections = []
        for result in deep_dive_results:
            if result.success and result.final_content:
                sections.append({
                    "topic": result.topic,
                    "content": result.final_content,
                    "low_confidence": result.low_confidence,
                })

        if not sections:
            step4.success = False
            step4.error = "No sections to synthesize"
            steps.append(step4)
            return PipelineResult(
                session_id=session_id,
                topic=topic,
                master_guide="",
                low_confidence_sections=0,
                total_sections=0,
                steps=steps,
                deep_dive_results=deep_dive_results,
                success=False,
                error="Step 4 failed: No sections to synthesize",
            )

        synthesis_response = self._claude.synthesize_guide(sections)

        step4.completed_at = datetime.now()
        step4.duration_ms = synthesis_response.duration_ms
        step4.success = synthesis_response.success
        step4.metadata = {
            "model": synthesis_response.model,
            "input_sections": len(sections),
            "output_length": len(synthesis_response.content),
        }

        if not synthesis_response.success:
            step4.error = synthesis_response.error
            steps.append(step4)
            return PipelineResult(
                session_id=session_id,
                topic=topic,
                master_guide="",
                low_confidence_sections=len(low_confidence),
                total_sections=len(sections),
                steps=steps,
                deep_dive_results=deep_dive_results,
                success=False,
                error=f"Step 4 failed: {synthesis_response.error}",
            )

        steps.append(step4)

        # Calculate total duration
        end_time = datetime.now()
        total_duration_ms = int((end_time - start_time).total_seconds() * 1000)

        self._log_status(f"Pipeline complete! Total duration: {total_duration_ms / 1000:.1f}s")

        return PipelineResult(
            session_id=session_id,
            topic=topic,
            master_guide=synthesis_response.content,
            low_confidence_sections=len(low_confidence),
            total_sections=len(sections),
            steps=steps,
            deep_dive_results=deep_dive_results,
            total_duration_ms=total_duration_ms,
            success=True,
        )
