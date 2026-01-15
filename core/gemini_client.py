"""Gemini API client for content generation.

Handles all interactions with Google's Gemini API, including retries
and timeout handling for production reliability.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

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
    BASE_KNOWLEDGE_PROMPT = """You are a world-class technical educator with deep expertise
in software engineering, distributed systems, and technology leadership.

Your task is to teach the following topic to a Principal Technical Program Manager
at a Mag7 company (Google, Meta, Apple, Amazon, Microsoft, Netflix, Nvidia).

Requirements:
- Assume the learner is highly technical but may be new to this specific domain
- Focus on strategic clarity AND technical depth
- Include real-world trade-offs and architectural considerations
- Cover what a Principal TPM needs to know to lead initiatives in this space
- Be comprehensive but avoid unnecessary fluff

Topic to teach:
{topic}

Provide a thorough, educational explanation."""

    SECTION_DRAFT_PROMPT = """You are a Principal Technical Program Manager at a Mag7 company
writing internal documentation for your engineering organization.

Write a comprehensive section on the following topic:
{topic}

Context from the overall guide:
{context}

Requirements:
- Write with the authority and depth expected at Principal level
- Include specific real-world trade-offs and their business implications
- Provide concrete examples where relevant
- Address both technical implementation and strategic considerations
- Be actionable - readers should be able to apply this knowledge

{feedback}

Write the section now."""

    SECTION_REWRITE_PROMPT = """You are a Principal Technical Program Manager at a Mag7 company.
Your previous draft was reviewed and needs improvement.

Original topic: {topic}

Your previous draft:
{previous_draft}

Reviewer feedback:
{critique}

Strictness level: {strictness}

Rewrite the section addressing ALL the feedback points. Maintain Principal TPM depth
and ensure the content would pass a Bar Raiser review at a Mag7 company."""

    def __init__(self) -> None:
        """Initialize Gemini client with API configuration."""
        self._settings = get_settings()
        self._model: Optional[genai.GenerativeModel] = None
        self._configure()

    def _configure(self) -> None:
        """Configure the Gemini API with credentials."""
        if not self._settings.is_gemini_configured():
            logger.warning("Gemini API key not configured")
            return

        genai.configure(api_key=self._settings.gemini_api_key)
        self._model = genai.GenerativeModel(
            model_name=self._settings.gemini_model,
            generation_config=GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=8192,
            ),
        )
        logger.info(f"Gemini client configured with model: {self._settings.gemini_model}")

    def is_available(self) -> bool:
        """Check if Gemini client is properly configured.

        Returns:
            True if client is ready to use.
        """
        return self._model is not None

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
            critique: Detailed critique from Claude.
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

        try:
            response = self._model.generate_content(
                prompt,
                request_options={"timeout": self._settings.api_timeout},
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
