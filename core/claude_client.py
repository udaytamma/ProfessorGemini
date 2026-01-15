"""Claude API client for structural planning, critique, and synthesis.

Handles all interactions with Anthropic's Claude API for the planning,
Bar Raiser critique, and final synthesis stages of the pipeline.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import anthropic

from config.settings import CriticStrictness, get_settings


logger = logging.getLogger(__name__)


@dataclass
class ClaudeResponse:
    """Response from Claude API call.

    Attributes:
        content: Generated text content.
        model: Model used for generation.
        duration_ms: Time taken in milliseconds.
        input_tokens: Input token count.
        output_tokens: Output token count.
        success: Whether the call succeeded.
        error: Error message if call failed.
    """

    content: str
    model: str
    duration_ms: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class CritiqueResult:
    """Result from Bar Raiser critique.

    Attributes:
        passed: Whether the draft passed review.
        feedback: Detailed feedback if failed.
        raw_response: Full response from Claude.
    """

    passed: bool
    feedback: str
    raw_response: str


class ClaudeClient:
    """Client for interacting with Anthropic Claude API.

    Provides methods for topic splitting, critique, and synthesis
    with automatic timeout handling.
    """

    # System prompt for topic splitting
    SPLIT_SYSTEM_PROMPT = """You are a technical content architect. Your task is to analyze
educational content and break it into logical, independent sub-topics that can be
explored in depth separately.

Each sub-topic should:
- Be self-contained enough to write a comprehensive section about
- Have clear boundaries that don't overlap significantly with other topics
- Be substantial enough to warrant deep exploration
- Together, cover all important aspects of the main topic

Return your response as a JSON array of topic strings, nothing else.
Example: ["Topic 1: Specific aspect", "Topic 2: Another aspect", ...]

Aim for 4-8 sub-topics depending on the complexity of the content."""

    # Critique prompts by strictness level
    CRITIQUE_PROMPTS = {
        CriticStrictness.HIGH: """You are a Mag7 Bar Raiser and VP of Engineering conducting
a rigorous review of technical documentation.

Review this draft with the highest standards:

DRAFT TO REVIEW:
{draft}

TOPIC: {topic}

Evaluate against these criteria:
1. SPECIFICITY: Does it include SPECIFIC real-world trade-offs with concrete examples?
   Not just "consider performance" but actual numbers, scenarios, case studies.

2. BUSINESS IMPACT: Is the business impact quantified or clearly articulated?
   Would an executive understand why this matters?

3. PRINCIPAL-LEVEL DEPTH: Would a Principal TPM at Google/Meta find this actionable?
   Does it go beyond what a senior engineer already knows?

4. STRATEGIC INSIGHT: Does it go beyond textbook definitions to strategic depth?
   Are there insights about when NOT to use something, or hidden gotchas?

5. COMPLETENESS: Are there obvious gaps that would leave a reader unprepared?

Your response must start with exactly "PASS" or "FAIL" on the first line.
If FAIL, list exactly what specific elements are missing or need improvement.
Be ruthlessly specific - vague feedback is not helpful.""",

        CriticStrictness.MEDIUM: """You are a senior technical reviewer evaluating
documentation quality.

Review this draft:

DRAFT TO REVIEW:
{draft}

TOPIC: {topic}

Check if this draft:
1. Covers the core concepts accurately and completely
2. Provides at least one meaningful trade-off or practical consideration
3. Is technically correct and internally coherent
4. Would be useful to a senior engineer learning this topic

Your response must start with exactly "PASS" or "FAIL" on the first line.
If FAIL, list the major gaps only. Don't nitpick minor issues.""",

        CriticStrictness.LOW: """You are a technical reviewer doing a quick sanity check.

Review this draft:

DRAFT TO REVIEW:
{draft}

TOPIC: {topic}

Check if this draft:
1. Is generally accurate (no major technical errors)
2. Covers the basic concepts
3. Is coherent and readable

Your response must start with exactly "PASS" or "FAIL" on the first line.
Only fail if there are significant errors or the content is clearly insufficient.""",
    }

    # Synthesis prompt
    SYNTHESIS_PROMPT = """You are a Principal Technical Program Manager creating a
comprehensive Master Guide from multiple sections.

Your task:
1. Review each section for Principal TPM depth - enhance if any section lacks
   the strategic insight or technical depth expected at Mag7 companies
2. Merge all sections into a cohesive Master Guide
3. Ensure consistent tone, terminology, and flow throughout
4. Add transitions between sections where needed
5. Create a logical structure with clear headings

Sections marked with [LOW CONFIDENCE] failed multiple review rounds - pay extra
attention to enhancing these.

SECTIONS TO SYNTHESIZE:
{sections}

Create the Master Guide now. Use markdown formatting with clear headers.
Start with a brief executive summary, then present each topic section."""

    def __init__(self) -> None:
        """Initialize Claude client with API configuration."""
        self._settings = get_settings()
        self._client: Optional[anthropic.Anthropic] = None
        self._configure()

    def _configure(self) -> None:
        """Configure the Claude API client."""
        if not self._settings.is_claude_configured():
            logger.warning("Anthropic API key not configured")
            return

        self._client = anthropic.Anthropic(
            api_key=self._settings.anthropic_api_key,
            timeout=self._settings.api_timeout,
        )
        logger.info(f"Claude client configured with model: {self._settings.claude_model}")

    def is_available(self) -> bool:
        """Check if Claude client is properly configured.

        Returns:
            True if client is ready to use.
        """
        return self._client is not None

    def split_into_topics(self, content: str) -> tuple[list[str], ClaudeResponse]:
        """Split base knowledge content into sub-topics.

        This is Step 2 of the pipeline.

        Args:
            content: The base knowledge content to split.

        Returns:
            Tuple of (list of topics, ClaudeResponse with metadata).
        """
        response = self._generate(
            system_prompt=self.SPLIT_SYSTEM_PROMPT,
            user_prompt=f"Analyze and split this content into sub-topics:\n\n{content}",
            operation="split_topics",
        )

        topics = []
        if response.success:
            try:
                # Parse JSON array from response
                # Handle potential markdown code blocks
                content_clean = response.content.strip()
                if content_clean.startswith("```"):
                    # Extract JSON from code block
                    lines = content_clean.split("\n")
                    content_clean = "\n".join(lines[1:-1])

                topics = json.loads(content_clean)
                if not isinstance(topics, list):
                    topics = [str(topics)]
                logger.info(f"Split content into {len(topics)} topics")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse topics JSON: {e}")
                # Fallback: try to extract topics from text
                topics = self._extract_topics_fallback(response.content)

        return topics, response

    def _extract_topics_fallback(self, content: str) -> list[str]:
        """Extract topics from non-JSON response.

        Args:
            content: Raw response content.

        Returns:
            List of extracted topics.
        """
        topics = []
        for line in content.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                # Clean up list markers
                topic = line.lstrip("-*0123456789.) ").strip()
                if topic:
                    topics.append(topic)
        return topics if topics else [content[:200]]

    def critique_draft(
        self,
        draft: str,
        topic: str,
        strictness: CriticStrictness,
    ) -> tuple[CritiqueResult, ClaudeResponse]:
        """Critique a draft section as Bar Raiser.

        This is part of Step 3 - the adversarial review loop.

        Args:
            draft: The draft content to review.
            topic: The topic being reviewed.
            strictness: How strict the review should be.

        Returns:
            Tuple of (CritiqueResult, ClaudeResponse with metadata).
        """
        prompt_template = self.CRITIQUE_PROMPTS[strictness]
        prompt = prompt_template.format(draft=draft, topic=topic)

        response = self._generate(
            system_prompt="You are a rigorous technical reviewer.",
            user_prompt=prompt,
            operation=f"critique:{strictness.value}",
        )

        # Parse PASS/FAIL from response
        passed = False
        feedback = ""

        if response.success and response.content:
            lines = response.content.strip().split("\n")
            first_line = lines[0].strip().upper()

            if first_line.startswith("PASS"):
                passed = True
                feedback = "Approved"
            else:
                passed = False
                feedback = "\n".join(lines[1:]).strip() if len(lines) > 1 else "No specific feedback"

        result = CritiqueResult(
            passed=passed,
            feedback=feedback,
            raw_response=response.content,
        )

        logger.info(f"Critique result for '{topic[:30]}...': {'PASS' if passed else 'FAIL'}")
        return result, response

    def synthesize_guide(
        self,
        sections: list[dict],
    ) -> ClaudeResponse:
        """Synthesize approved sections into a Master Guide.

        This is Step 4 of the pipeline.

        Args:
            sections: List of dicts with 'topic', 'content', and 'low_confidence' keys.

        Returns:
            ClaudeResponse with the synthesized Master Guide.
        """
        # Format sections for synthesis
        formatted_sections = []
        for section in sections:
            marker = "[LOW CONFIDENCE] " if section.get("low_confidence") else ""
            formatted_sections.append(
                f"## {marker}{section['topic']}\n\n{section['content']}"
            )

        sections_text = "\n\n---\n\n".join(formatted_sections)

        prompt = self.SYNTHESIS_PROMPT.format(sections=sections_text)

        return self._generate(
            system_prompt="You are a Principal TPM creating authoritative technical documentation.",
            user_prompt=prompt,
            operation="synthesis",
            max_tokens=16000,  # Allow longer output for full guide
        )

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        operation: str,
        max_tokens: int = 4096,
    ) -> ClaudeResponse:
        """Execute generation request with timing and error handling.

        Args:
            system_prompt: System context for Claude.
            user_prompt: The user message/prompt.
            operation: Name of the operation for logging.
            max_tokens: Maximum tokens in response.

        Returns:
            ClaudeResponse with results or error.
        """
        if not self.is_available():
            return ClaudeResponse(
                content="",
                model=self._settings.claude_model,
                duration_ms=0,
                success=False,
                error="Claude client not configured",
            )

        start_time = time.time()

        try:
            response = self._client.messages.create(
                model=self._settings.claude_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # Extract text content
            content = ""
            if response.content:
                content = response.content[0].text

            logger.info(f"Claude {operation} completed in {duration_ms}ms")

            return ClaudeResponse(
                content=content,
                model=self._settings.claude_model,
                duration_ms=duration_ms,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                success=True,
            )

        except anthropic.APITimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = f"API timeout after {self._settings.api_timeout}s"
            logger.error(f"Claude {operation} timeout: {error_msg}")

            return ClaudeResponse(
                content="",
                model=self._settings.claude_model,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )

        except anthropic.APIError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"Claude {operation} API error: {error_msg}")

            return ClaudeResponse(
                content="",
                model=self._settings.claude_model,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"Claude {operation} failed: {error_msg}")

            return ClaudeResponse(
                content="",
                model=self._settings.claude_model,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )
