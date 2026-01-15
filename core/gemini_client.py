"""Gemini API client for content generation.

Handles all interactions with Google's Gemini API using the new google.genai SDK.
Includes retries and timeout handling for production reliability.
Supports both sync and async operations for optimal performance.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Response from Gemini API call.

    Attributes:
        content: Generated text content.
        model: Model used for generation.
        duration_ms: Time taken in milliseconds.
        token_count: Approximate token count (if available).
        success: Whether the call succeeded.
        error: Error message if call failed.
    """

    content: str
    model: str
    duration_ms: int
    token_count: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


class GeminiClient:
    """Client for interacting with Google Gemini API.

    Provides methods for content generation with automatic retry
    and timeout handling.
    """

    # System prompts for different pipeline stages
    BASE_KNOWLEDGE_PROMPT = """You are a world-class technical educator with 20+ years of experience at companies like Google, Amazon, and Netflix.
Your audience is preparing for Principal Technical Program Manager job interviews at Mag7 companies.

Create a comprehensive technical deep-dive on the topic that a Principal TPM at a Mag7 company would need to know.

When you answer, ensure your coverage includes:
(1) Real-world behavior/examples at a Mag7
(2) Tradeoffs for every choice made or action taken
(3) Impact on business/ROI/CX/Skill/Business capabilities

**CRITICAL FORMATTING REQUIREMENT:**
You MUST structure your response with sections using Roman numeral headers (e.g., I, II, III, IV, V).
Each section MUST start with "## I. ", "## II. ", etc. on its own line.
This format is MANDATORY for proper parsing. Do NOT use any other numbering system.

TOPIC: {topic}"""

    SECTION_DRAFT_PROMPT = """Create a comprehensive technical deep-dive on the topic that a Principal TPM at a Mag7 company would need to know.

When you answer, ensure your coverage includes:
(1) Real-world behavior/examples at a Mag7
(2) Tradeoffs for every choice made or action taken
(3) Impact on business/ROI/CX/Skill/Business capabilities

TOPIC: {topic}

CONTEXT FROM OVERALL GUIDE:
{context}

Write a comprehensive section that includes:
- **Technical depth**: Explain the "how" and "why", not just the "what"
- **Real-world examples**: Reference specific implementations at Google, Amazon, Netflix, etc.
- **Tradeoffs analysis**: Discuss pros/cons with quantified impacts where possible
- **Actionable guidance**: Readers should know exactly what to do after reading
- **Edge cases**: Address common failure modes and how to handle them
{feedback}

**INTERVIEW QUESTIONS:**
At the end of your response, include a section titled "## Interview Questions" with at least 2 challenging interview questions that a Principal TPM might be asked about this topic. Include brief guidance on what a strong answer should cover.

TARGET: A substantive deep-dive, Principal-level content. No fluff or generic statements. No metaphors, strictly professional."""

    SECTION_REWRITE_PROMPT = """Your previous draft was reviewed by a Mag7 Bar Raiser and needs improvement.

TOPIC: {topic}
REVIEW STRICTNESS: {strictness}

YOUR PREVIOUS DRAFT:
{previous_draft}

BAR RAISER FEEDBACK:
{critique}

REWRITE INSTRUCTIONS:
1. Address ALL feedback points explicitly
2. Add specific examples, numbers, or case studies where feedback indicates gaps
3. Strengthen trade-off analysis with concrete comparisons
4. Ensure every paragraph adds unique value (no repetition or filler)
5. Maintain or increase technical depth

Output the improved section directly. Target 600-900 words."""

    # Prompts for Gemini-only mode (when not using Claude)
    SPLIT_TOPICS_PROMPT = """Analyze this technical content and identify distinct sub-topics (by Roman numerals) that warrant deep-dive and further exploration.

CONTENT:
{content}

CRITERIA FOR GOOD SUB-TOPICS:
- Each should be substantial enough for a deep-dive
- Topics should be complementary but not overlapping
- Include both foundational concepts AND advanced/operational aspects

Return ONLY a JSON array of descriptive topic strings.
Example: ["Topic 1: Core Architecture and Component Design", "Topic 2: Scaling Strategies and Performance Optimization"]

No other text, just the JSON array."""

    CRITIQUE_PROMPTS = {
        "high": """You are a VP of Engineering at Google conducting a Bar Raiser review.

TOPIC: {topic}

DRAFT TO REVIEW:
{draft}

EVALUATION CRITERIA (all must pass):
1. SPECIFICITY: Contains specific examples, numbers, or case studies (not generic statements)
2. TRADE-OFFS: Discusses concrete pros/cons with quantified or comparative analysis
3. ACTIONABILITY: Reader knows exactly what to do after reading
4. DEPTH: Goes beyond surface-level explanation to expert-level insight
5. COMPLETENESS: No obvious gaps that would leave a Principal TPM unprepared

First line must be exactly: PASS or FAIL
If FAIL, list 2-3 specific improvements needed (be actionable, not vague).""",

        "medium": """Senior technical review for documentation quality.

TOPIC: {topic}

DRAFT:
{draft}

CHECK:
1. Technical accuracy - no errors or misconceptions
2. Has at least one concrete trade-off discussion
3. Provides actionable guidance
4. Appropriate depth for senior engineers

First line: PASS or FAIL
If FAIL: List specific issues to fix.""",

        "low": """Quick quality check.

TOPIC: {topic}

DRAFT:
{draft}

PASS if: (1) No major technical errors (2) Covers the topic basics (3) Readable and coherent

First line: PASS or FAIL
Only FAIL for significant errors or clearly insufficient content.""",
    }

    SYNTHESIS_PROMPT = """You are creating a comprehensive Master Guide for Principal TPMs from multiple expert-written sections.

SECTIONS TO SYNTHESIZE:
{sections}

YOUR TASKS:
1. **Unify**: Merge sections into a cohesive narrative with consistent terminology and tone
2. **Enhance**: Add smooth transitions between sections that show how concepts connect
3. **Quality Check**: Sections marked [LOW CONFIDENCE] may need strengthening - improve them
4. **Structure**: Create a logical flow from fundamentals to advanced topics

OUTPUT FORMAT:
- Title: Use the main topic only (e.g., "# Load Balancing" not "# The Principal TPM's Guide to...")
- Executive Summary: 3-4 sentences covering what the reader will learn and why it matters
- Sections: Use ## headers, maintain depth, include all key content from source sections
- Conclusion: Brief wrap-up with key takeaways for a Principal TPM

Preserve technical depth and specific examples from the source sections. The final guide should be comprehensive and immediately actionable."""

    def __init__(self) -> None:
        """Initialize Gemini client with API configuration."""
        self._settings = get_settings()
        self._client: Optional[genai.Client] = None
        self._configure()

    def _configure(self) -> None:
        """Configure the Gemini API with credentials and optimized connection pooling."""
        if not self._settings.is_gemini_configured():
            logger.warning("Gemini API key not configured")
            return

        # Configure HTTP options for better connection reuse
        # Increase timeout and connection limits for parallel operations
        http_options = types.HttpOptions(
            timeout=120000,  # 120 seconds for long operations
        )

        self._client = genai.Client(
            api_key=self._settings.gemini_api_key,
            http_options=http_options,
        )
        logger.info(f"Gemini client configured with model: {self._settings.gemini_model}")

    def is_available(self) -> bool:
        """Check if Gemini client is properly configured.

        Returns:
            True if client is ready to use.
        """
        return self._client is not None

    def generate_base_knowledge(self, topic: str) -> GeminiResponse:
        """Generate base knowledge content for a topic.

        This is Step 1 of the pipeline - creating comprehensive
        foundational content about the topic.

        Args:
            topic: The topic to generate content for.

        Returns:
            GeminiResponse with generated content.
        """
        prompt = self.BASE_KNOWLEDGE_PROMPT.format(topic=topic)
        return self._generate(prompt, "base_knowledge")

    def generate_section_draft(
        self,
        topic: str,
        context: str,
        feedback: str = "",
    ) -> GeminiResponse:
        """Generate a draft section for a specific topic.

        Used in Step 3 of the pipeline for parallel deep dives.

        Args:
            topic: The specific sub-topic to write about.
            context: Context from the base knowledge.
            feedback: Optional feedback from previous attempts.

        Returns:
            GeminiResponse with draft content.
        """
        feedback_section = ""
        if feedback:
            feedback_section = f"\nPrevious feedback to address:\n{feedback}"

        prompt = self.SECTION_DRAFT_PROMPT.format(
            topic=topic,
            context=context[:2000],  # Limit context to avoid token overflow
            feedback=feedback_section,
        )
        return self._generate(prompt, f"section_draft:{topic[:50]}")

    def rewrite_section(
        self,
        topic: str,
        previous_draft: str,
        critique: str,
        strictness: str,
    ) -> GeminiResponse:
        """Rewrite a section based on critique feedback.

        Called when the Bar Raiser rejects a draft.

        Args:
            topic: The topic being written.
            previous_draft: The rejected draft.
            critique: Detailed critique from reviewer.
            strictness: Current strictness level.

        Returns:
            GeminiResponse with rewritten content.
        """
        prompt = self.SECTION_REWRITE_PROMPT.format(
            topic=topic,
            previous_draft=previous_draft[:4000],  # Limit to avoid overflow
            critique=critique,
            strictness=strictness,
        )
        return self._generate(prompt, f"section_rewrite:{topic[:50]}")

    def split_into_topics(self, content: str) -> tuple[list[str], GeminiResponse]:
        """Split base knowledge content into sub-topics.

        Used in Gemini-only mode for Step 2 of the pipeline.

        Args:
            content: The base knowledge content to split.

        Returns:
            Tuple of (list of topics, GeminiResponse with metadata).
        """
        prompt = self.SPLIT_TOPICS_PROMPT.format(content=content)
        response = self._generate(prompt, "split_topics")

        topics = []
        if response.success:
            try:
                # Parse JSON array from response
                content_clean = response.content.strip()
                if content_clean.startswith("```"):
                    lines = content_clean.split("\n")
                    content_clean = "\n".join(lines[1:-1])

                topics = json.loads(content_clean)
                if not isinstance(topics, list):
                    topics = [str(topics)]
                logger.info(f"Split content into {len(topics)} topics")
            except json.JSONDecodeError:
                # Fallback: extract topics from text
                topics = self._extract_topics_fallback(response.content)

        return topics, response

    def _extract_topics_fallback(self, content: str) -> list[str]:
        """Extract topics from non-JSON response."""
        topics = []
        for line in content.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                topic = line.lstrip("-*0123456789.) ").strip()
                if topic:
                    topics.append(topic)
        return topics if topics else [content[:200]]

    def critique_draft(
        self,
        draft: str,
        topic: str,
        strictness: str,
    ) -> tuple[dict, GeminiResponse]:
        """Critique a draft section as Bar Raiser.

        Used in Gemini-only mode for the adversarial review loop.

        Args:
            draft: The draft content to review.
            topic: The topic being reviewed.
            strictness: Strictness level (high, medium, low).

        Returns:
            Tuple of (CritiqueResult dict, GeminiResponse with metadata).
        """
        prompt_template = self.CRITIQUE_PROMPTS.get(strictness, self.CRITIQUE_PROMPTS["medium"])
        prompt = prompt_template.format(draft=draft, topic=topic)

        response = self._generate(prompt, f"critique:{strictness}")

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

        result = {
            "passed": passed,
            "feedback": feedback,
            "raw_response": response.content,
        }

        logger.info(f"Critique result for '{topic[:30]}...': {'PASS' if passed else 'FAIL'}")
        return result, response

    def synthesize_guide(self, sections: list[dict]) -> GeminiResponse:
        """Synthesize approved sections into a Master Guide.

        Used in Gemini-only mode for Step 4 of the pipeline.

        Args:
            sections: List of dicts with 'topic', 'content', and 'low_confidence' keys.

        Returns:
            GeminiResponse with the synthesized Master Guide.
        """
        formatted_sections = []
        for section in sections:
            marker = "[LOW CONFIDENCE] " if section.get("low_confidence") else ""
            formatted_sections.append(
                f"## {marker}{section['topic']}\n\n{section['content']}"
            )

        sections_text = "\n\n---\n\n".join(formatted_sections)
        prompt = self.SYNTHESIS_PROMPT.format(sections=sections_text)

        return self._generate(prompt, "synthesis")

    # Token limits by operation type for efficiency
    # None = no limit (use model default)
    # Note: Previously had critique: 512, but Gemini was returning empty responses
    # Letting the model decide output length works better
    TOKEN_LIMITS = {
        # All operations use model default (no artificial limits)
    }

    def _get_token_limit(self, operation: str) -> Optional[int]:
        """Get appropriate token limit for operation type.

        Returns None to use model default (no limit).
        """
        for key, limit in self.TOKEN_LIMITS.items():
            if key in operation:
                return limit
        return None  # Default: no limit

    def _generate(self, prompt: str, operation: str) -> GeminiResponse:
        """Execute generation request with timing and error handling.

        Args:
            prompt: The prompt to send to Gemini.
            operation: Name of the operation for logging.

        Returns:
            GeminiResponse with results or error.
        """
        if not self.is_available():
            return GeminiResponse(
                content="",
                model=self._settings.gemini_model,
                duration_ms=0,
                success=False,
                error="Gemini client not configured",
            )

        start_time = time.time()
        max_tokens = self._get_token_limit(operation)

        # Build config - only include max_output_tokens if set
        config_kwargs = {"temperature": 0.7, "top_p": 0.95}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        try:
            response = self._client.models.generate_content(
                model=self._settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # Extract text content
            if response.text:
                content = response.text
            else:
                # Handle blocked or empty responses
                content = ""
                logger.warning(f"Empty response from Gemini for {operation}")

            logger.info(f"Gemini {operation} completed in {duration_ms}ms")

            return GeminiResponse(
                content=content,
                model=self._settings.gemini_model,
                duration_ms=duration_ms,
                success=True,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"Gemini {operation} failed: {error_msg}")

            return GeminiResponse(
                content="",
                model=self._settings.gemini_model,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )

    # ============== ASYNC METHODS ==============

    async def _generate_async(self, prompt: str, operation: str) -> GeminiResponse:
        """Execute async generation request with timing and error handling.

        Args:
            prompt: The prompt to send to Gemini.
            operation: Name of the operation for logging.

        Returns:
            GeminiResponse with results or error.
        """
        if not self.is_available():
            return GeminiResponse(
                content="",
                model=self._settings.gemini_model,
                duration_ms=0,
                success=False,
                error="Gemini client not configured",
            )

        start_time = time.time()
        max_tokens = self._get_token_limit(operation)

        # Build config - only include max_output_tokens if set
        config_kwargs = {"temperature": 0.7, "top_p": 0.95}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        try:
            # Use async client via aio namespace
            response = await self._client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            duration_ms = int((time.time() - start_time) * 1000)

            if response.text:
                content = response.text
            else:
                content = ""
                logger.warning(f"Empty response from Gemini for {operation}")

            logger.info(f"Gemini {operation} completed in {duration_ms}ms (async)")

            return GeminiResponse(
                content=content,
                model=self._settings.gemini_model,
                duration_ms=duration_ms,
                success=True,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"Gemini {operation} failed: {error_msg}")

            return GeminiResponse(
                content="",
                model=self._settings.gemini_model,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )

    async def generate_section_draft_async(
        self,
        topic: str,
        context: str,
        feedback: str = "",
    ) -> GeminiResponse:
        """Async version of generate_section_draft."""
        feedback_section = ""
        if feedback:
            feedback_section = f"\nPrevious feedback to address:\n{feedback}"

        prompt = self.SECTION_DRAFT_PROMPT.format(
            topic=topic,
            context=context[:2000],
            feedback=feedback_section,
        )
        return await self._generate_async(prompt, f"section_draft:{topic[:50]}")

    async def rewrite_section_async(
        self,
        topic: str,
        previous_draft: str,
        critique: str,
        strictness: str,
    ) -> GeminiResponse:
        """Async version of rewrite_section."""
        prompt = self.SECTION_REWRITE_PROMPT.format(
            topic=topic,
            previous_draft=previous_draft[:4000],
            critique=critique,
            strictness=strictness,
        )
        return await self._generate_async(prompt, f"section_rewrite:{topic[:50]}")

    async def critique_draft_async(
        self,
        draft: str,
        topic: str,
        strictness: str,
    ) -> tuple[dict, GeminiResponse]:
        """Async version of critique_draft."""
        prompt_template = self.CRITIQUE_PROMPTS.get(strictness, self.CRITIQUE_PROMPTS["medium"])
        prompt = prompt_template.format(draft=draft, topic=topic)

        response = await self._generate_async(prompt, f"critique:{strictness}")

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

        result = {
            "passed": passed,
            "feedback": feedback,
            "raw_response": response.content,
        }

        logger.info(f"Critique result for '{topic[:30]}...': {'PASS' if passed else 'FAIL'}")
        return result, response
