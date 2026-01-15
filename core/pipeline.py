"""Main pipeline orchestration for Professor Gemini.

Coordinates the 4-step hybrid AI pipeline:
1. Base Knowledge (Gemini) - with Roman numeral sections
2. Topic Split (Local parsing of Roman numerals OR Gemini/Claude)
3. Deep Dive (Gemini drafts, optional Bar Raiser critique)
4. Synthesis (Local concatenation OR Gemini/Claude)

Supports multiple optimization modes to reduce Gemini API calls:
- Local topic split: Parse Roman numerals instead of calling Gemini
- Critique disabled: Accept drafts without review loop
- Local synthesis: Concatenate sections instead of calling Gemini
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
from uuid import uuid4

from config.settings import get_settings
from core.gemini_client import GeminiClient
from core.bar_raiser import BarRaiser, BarRaiserResult
from core.local_processing import split_by_roman_numerals, synthesize_locally


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

    Optimization modes (controlled by settings):
    - enable_critique=False: Skip Bar Raiser review loop
    - local_synthesis=True: Use local concatenation instead of Gemini
    - Local topic split: Always uses Roman numeral parsing first
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
        self._claude = None

        # Only initialize Claude if needed
        if self._settings.use_claude:
            from core.claude_client import ClaudeClient
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

        if self._settings.use_claude and (self._claude is None or not self._claude.is_available()):
            issues.append("Claude API not configured (required when USE_CLAUDE=true)")

        if issues:
            return False, "; ".join(issues)

        # Build mode description
        modes = []
        if self._settings.use_claude:
            modes.append("Hybrid (Gemini + Claude)")
        else:
            modes.append("Gemini-only")
        if not self._settings.enable_critique:
            modes.append("critique OFF")
        if self._settings.local_synthesis:
            modes.append("local synthesis")

        return True, f"Pipeline ready ({', '.join(modes)})"

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

        # Step 1: Base Knowledge (always uses Gemini)
        self._log_status("1. Generating base knowledge with Gemini...")
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
            self._log_status("1. Generating base knowledge with Gemini... FAILED")
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
        self._log_status("1. Generating base knowledge with Gemini... completed")

        # Step 2: Split into topics (LOCAL first, fallback to Gemini/Claude)
        self._log_status("2. Splitting response into sections...")
        step2 = PipelineStep(name="split_topics", started_at=datetime.now())

        # Try local Roman numeral parsing first (no API call)
        local_split = split_by_roman_numerals(base_response.content)

        if local_split.success and len(local_split.topics) >= 3:
            # Local split succeeded - use it
            sub_topics = local_split.topics
            section_content_map = local_split.sections
            step2.metadata["method"] = "local_roman_numerals"
            step2.metadata["model"] = "local"
            step2.duration_ms = 0
            step2.success = True
            logger.info(f"Local split succeeded: {len(sub_topics)} sections")
        else:
            # Fallback to Gemini/Claude API call
            logger.warning("Local split failed or insufficient sections, falling back to API")
            section_content_map = {}  # No pre-extracted content

            if self._settings.use_claude:
                sub_topics, split_response = self._claude.split_into_topics(base_response.content)
                step2.metadata["method"] = "claude_api"
                step2.metadata["model"] = split_response.model
                step2.duration_ms = split_response.duration_ms
                step2.success = split_response.success and len(sub_topics) > 0
                if not step2.success:
                    step2.error = split_response.error or "No topics extracted"
            else:
                sub_topics, split_response = self._gemini.split_into_topics(base_response.content)
                step2.metadata["method"] = "gemini_api"
                step2.metadata["model"] = split_response.model
                step2.duration_ms = split_response.duration_ms
                step2.success = split_response.success and len(sub_topics) > 0
                if not step2.success:
                    step2.error = split_response.error or "No topics extracted"

        step2.completed_at = datetime.now()
        step2.metadata["topic_count"] = len(sub_topics)
        step2.metadata["topics"] = sub_topics

        if not step2.success:
            steps.append(step2)
            self._log_status("2. Splitting response into sections... FAILED")
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
        self._log_status(f"2. Splitting response into sections... completed ({len(sub_topics)} sections)")

        # Step 3: Deep dive (with or without Bar Raiser critique)
        if self._settings.enable_critique:
            self._log_status("3. Generating deep dives with critique...")
        else:
            self._log_status("3. Generating deep dives...")

        step3 = PipelineStep(name="deep_dive", started_at=datetime.now())

        # Check if we already have content from local split
        if section_content_map:
            # We have pre-extracted content from Roman numeral split
            # Still generate deeper content for each section
            deep_dive_results = self._generate_deep_dives_from_sections(
                topics=sub_topics,
                section_content=section_content_map,
                context=base_response.content,
            )
        else:
            # No pre-extracted content - use Bar Raiser (with or without critique)
            bar_raiser = BarRaiser(
                gemini_client=self._gemini,
                claude_client=self._claude,
                status_callback=None,
            )

            if self._settings.use_claude:
                with ThreadPoolExecutor(max_workers=self._settings.max_workers) as executor:
                    deep_dive_results = bar_raiser.process_topics_parallel(
                        topics=sub_topics,
                        context=base_response.content,
                        executor=executor,
                    )
            else:
                deep_dive_results = asyncio.run(
                    bar_raiser.process_topics_async(
                        topics=sub_topics,
                        context=base_response.content,
                    )
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
            "critique_enabled": self._settings.enable_critique,
        }

        steps.append(step3)
        self._log_status(f"3. Generating deep dives... completed ({len(successful)}/{len(sub_topics)} sections)")

        # Step 4: Synthesis (local or API)
        self._log_status("4. Collating final response...")
        step4 = PipelineStep(name="synthesis", started_at=datetime.now())

        # Prepare sections for synthesis (maintain order from split)
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
            self._log_status("4. Collating final response... FAILED (no content)")
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

        # Choose synthesis method based on settings
        if self._settings.local_synthesis:
            # Local synthesis - no API call
            synthesis_result = synthesize_locally(sections, topic)
            step4.metadata["method"] = "local"
            step4.metadata["model"] = "local"
            step4.duration_ms = 0
            step4.success = synthesis_result.success

            if synthesis_result.success:
                master_guide = synthesis_result.content
            else:
                step4.error = synthesis_result.error
                master_guide = ""
        else:
            # API synthesis
            if self._settings.use_claude:
                synthesis_response = self._claude.synthesize_guide(sections)
            else:
                synthesis_response = self._gemini.synthesize_guide(sections)

            step4.metadata["method"] = "claude_api" if self._settings.use_claude else "gemini_api"
            step4.metadata["model"] = synthesis_response.model
            step4.duration_ms = synthesis_response.duration_ms
            step4.success = synthesis_response.success

            if synthesis_response.success:
                master_guide = synthesis_response.content
            else:
                step4.error = synthesis_response.error
                master_guide = ""

        step4.completed_at = datetime.now()
        step4.metadata["input_sections"] = len(sections)
        step4.metadata["output_length"] = len(master_guide)

        if not step4.success:
            steps.append(step4)
            self._log_status("4. Collating final response... FAILED")
            return PipelineResult(
                session_id=session_id,
                topic=topic,
                master_guide="",
                low_confidence_sections=len(low_confidence),
                total_sections=len(sections),
                steps=steps,
                deep_dive_results=deep_dive_results,
                success=False,
                error=f"Step 4 failed: {step4.error}",
            )

        steps.append(step4)
        self._log_status("4. Collating final response... completed")

        # Calculate total duration
        end_time = datetime.now()
        total_duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return PipelineResult(
            session_id=session_id,
            topic=topic,
            master_guide=master_guide,
            low_confidence_sections=len(low_confidence),
            total_sections=len(sections),
            steps=steps,
            deep_dive_results=deep_dive_results,
            total_duration_ms=total_duration_ms,
            success=True,
        )

    def _generate_deep_dives_from_sections(
        self,
        topics: list[str],
        section_content: dict[str, str],
        context: str,
    ) -> list[BarRaiserResult]:
        """Generate deep dives using pre-extracted section content.

        When local Roman numeral split succeeded, we already have content
        for each section. We can either:
        - Use it directly (fastest, 0 API calls)
        - Generate enhanced drafts with Gemini (better quality, N API calls)

        Currently: Uses pre-extracted content directly for maximum efficiency.

        Args:
            topics: List of topic titles from split.
            section_content: Map of topic title to section content.
            context: Full base knowledge content for reference.

        Returns:
            List of BarRaiserResult with section content.
        """
        results = []

        for topic in topics:
            content = section_content.get(topic, "")

            if content:
                # Use pre-extracted content directly (no API call)
                results.append(BarRaiserResult(
                    topic=topic,
                    final_content=content,
                    low_confidence=False,  # Content from base knowledge is trusted
                    attempts=[],
                    total_duration_ms=0,
                    success=True,
                ))
            else:
                # Missing content - mark as failed
                logger.warning(f"No content found for topic: {topic}")
                results.append(BarRaiserResult(
                    topic=topic,
                    final_content="",
                    low_confidence=True,
                    attempts=[],
                    total_duration_ms=0,
                    success=False,
                    error=f"No content extracted for topic: {topic}",
                ))

        return results
