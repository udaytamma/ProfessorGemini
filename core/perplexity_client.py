"""Perplexity API client for web search with AI synthesis.

Uses Perplexity's OpenAI-compatible API to perform web searches
and return AI-synthesized responses.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, APITimeoutError, APIError

from config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class PerplexityResponse:
    """Response from Perplexity API call.

    Attributes:
        content: Generated text content with web search results.
        model: Model used for generation.
        duration_ms: Time taken in milliseconds.
        success: Whether the call succeeded.
        error: Error message if call failed.
    """

    content: str
    model: str
    duration_ms: int
    success: bool = True
    error: Optional[str] = None


class PerplexityClient:
    """Client for interacting with Perplexity API.

    Uses the OpenAI-compatible API with Perplexity's sonar-pro model
    for web search with AI synthesis.
    """

    BASE_URL = "https://api.perplexity.ai"

    def __init__(self) -> None:
        """Initialize Perplexity client with API configuration."""
        self._settings = get_settings()
        self._client: Optional[OpenAI] = None
        self._configure()

    def _configure(self) -> None:
        """Configure the Perplexity API client."""
        if not self._settings.is_perplexity_configured():
            logger.warning("Perplexity API key not configured")
            return

        self._client = OpenAI(
            api_key=self._settings.perplexity_api_key,
            base_url=self.BASE_URL,
            timeout=self._settings.api_timeout,
        )
        logger.info(f"Perplexity client configured with model: {self._settings.perplexity_model}")

    def is_available(self) -> bool:
        """Check if Perplexity client is properly configured.

        Returns:
            True if client is ready to use.
        """
        return self._client is not None

    @staticmethod
    def _strip_citations(content: str) -> str:
        """Remove citation markers like [1], [2][3], etc. from content.

        Args:
            content: Text with citation markers.

        Returns:
            Text with citation markers removed.
        """
        # Remove patterns like [1], [2][3], [1][2][3][4][5], etc.
        return re.sub(r'\[\d+\]', '', content)

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful research assistant. Provide comprehensive, "
        "well-structured responses with relevant information from the web. "
        "Use markdown formatting for clarity."
    )

    def search(self, query: str, system_prompt: Optional[str] = None) -> PerplexityResponse:
        """Execute a web search query using Perplexity.

        Args:
            query: The search query or prompt.
            system_prompt: Optional custom system prompt. If not provided, uses default.

        Returns:
            PerplexityResponse with search results or error.
        """
        if not self.is_available():
            return PerplexityResponse(
                content="",
                model=self._settings.perplexity_model,
                duration_ms=0,
                success=False,
                error="Perplexity client not configured",
            )

        start_time = time.time()
        prompt = system_prompt.strip() if system_prompt and system_prompt.strip() else self.DEFAULT_SYSTEM_PROMPT

        try:
            response = self._client.chat.completions.create(
                model=self._settings.perplexity_model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                extra_body={
                    "return_citations": False,  # Disable citations to save tokens/cost
                },
            )

            duration_ms = int((time.time() - start_time) * 1000)

            content = ""
            if response.choices and response.choices[0].message:
                raw_content = response.choices[0].message.content or ""
                content = self._strip_citations(raw_content)

            logger.info(f"Perplexity search completed in {duration_ms}ms")

            return PerplexityResponse(
                content=content,
                model=self._settings.perplexity_model,
                duration_ms=duration_ms,
                success=True,
            )

        except APITimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = f"API timeout after {self._settings.api_timeout}s"
            logger.error(f"Perplexity search timeout: {error_msg}")

            return PerplexityResponse(
                content="",
                model=self._settings.perplexity_model,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )

        except APIError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"Perplexity search API error: {error_msg}")

            return PerplexityResponse(
                content="",
                model=self._settings.perplexity_model,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"Perplexity search failed: {error_msg}")

            return PerplexityResponse(
                content="",
                model=self._settings.perplexity_model,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            )
