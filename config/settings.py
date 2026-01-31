"""Application settings with environment variable loading.

Follows the same pattern as IngredientScanner for consistency across projects.
Supports both local development (.env) and cloud deployment (environment variables).
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


load_dotenv()


class CriticStrictness(str, Enum):
    """Critic strictness levels for the Bar Raiser loop."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Attributes:
        gemini_api_key: Google Gemini API key.
        gemini_model: Gemini model identifier.
        anthropic_api_key: Anthropic Claude API key.
        claude_model: Claude model identifier.
        cyrus_root_path: Path to Cyrus project root for Nebula integration.
        default_strictness: Default critic strictness level.
        max_workers: Maximum parallel workers for deep dive.
        max_retries: Maximum retry attempts per topic.
        api_timeout: Timeout for API calls in seconds.
        log_retention_days: Days to retain request history.
        app_port: Streamlit server port.
        app_host: Streamlit server host.
        log_level: Application logging level.
    """

    # Gemini Configuration
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3-pro-preview")

    # Claude Configuration (optional - only used if use_claude=True)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-opus-4-5-20251101")

    # Mode Selection: Use Claude for critique/synthesis or Gemini-only
    use_claude: bool = Field(default=False, alias="USE_CLAUDE")

    # Cyrus Integration
    cyrus_root_path: str = Field(
        default="/Users/omega/Projects/Cyrus",
        description="Path to Cyrus project root for Nebula integration",
    )

    # Qdrant Configuration (reuse IngredientScanner cluster)
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")

    # RAG Configuration
    rag_enabled: bool = Field(
        default=True,
        alias="RAG_ENABLED",
        description="Use RAG for semantic retrieval instead of full context",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        alias="RAG_TOP_K",
        description="Number of documents to retrieve for RAG",
    )
    rag_collection: str = Field(
        default="professor_gemini",
        description="Qdrant collection name for RAG documents",
    )

    # Pipeline Configuration
    default_strictness: CriticStrictness = Field(default=CriticStrictness.LOW)
    max_workers: int = Field(default=10, ge=1, le=20)
    max_retries: int = Field(default=2, ge=1, le=5)
    api_timeout: int = Field(default=120, ge=30, le=300)

    # Optimization Settings (to reduce Gemini API calls)
    enable_critique: bool = Field(
        default=False,
        alias="ENABLE_CRITIQUE",
        description="Enable Bar Raiser critique loop. When OFF, drafts are accepted without review.",
    )
    local_synthesis: bool = Field(
        default=True,
        alias="LOCAL_SYNTHESIS",
        description="Use local synthesis (concatenate sections) instead of Gemini synthesis call.",
    )

    # Logging Configuration
    log_retention_days: int = Field(default=30, ge=1, le=365)
    log_level: str = Field(default="INFO")

    # Server Configuration (for cloud deployment)
    app_port: int = Field(default=8502, ge=1024, le=65535)
    app_host: str = Field(default="0.0.0.0")

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Perplexity Configuration
    perplexity_api_key: str = Field(default="", alias="PERPLEXITY_API_KEY")
    perplexity_model: str = Field(default="sonar-pro")

    @field_validator("gemini_api_key", "anthropic_api_key", "qdrant_api_key", "perplexity_api_key", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip whitespace from API keys."""
        return v.strip() if v else ""

    def is_gemini_configured(self) -> bool:
        """Check if Gemini API is configured.

        Returns:
            True if Gemini API key is set.
        """
        return bool(self.gemini_api_key)

    def is_claude_configured(self) -> bool:
        """Check if Claude API is configured.

        Returns:
            True if Anthropic API key is set.
        """
        return bool(self.anthropic_api_key)

    def is_fully_configured(self) -> bool:
        """Check if all required APIs are configured based on mode.

        Returns:
            True if required APIs are configured for the selected mode.
        """
        if self.use_claude:
            return self.is_gemini_configured() and self.is_claude_configured()
        return self.is_gemini_configured()

    def is_qdrant_configured(self) -> bool:
        """Check if Qdrant is configured for RAG.

        Returns:
            True if Qdrant URL and API key are set.
        """
        return bool(self.qdrant_url and self.qdrant_api_key)

    def is_rag_available(self) -> bool:
        """Check if RAG is enabled and Qdrant is configured.

        Returns:
            True if RAG can be used.
        """
        return self.rag_enabled and self.is_qdrant_configured()

    def is_perplexity_configured(self) -> bool:
        """Check if Perplexity API is configured.

        Returns:
            True if Perplexity API key is set.
        """
        return bool(self.perplexity_api_key)

    def get_gemini_responses_path(self) -> str:
        """Get the path to gemini-responses directory in Cyrus.

        Returns:
            Full path to gemini-responses directory.
        """
        return os.path.join(self.cyrus_root_path, "gemini-responses")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance with loaded configuration.
    """
    return Settings()


def refresh_settings() -> Settings:
    """Clear settings cache and reload.

    Useful for testing or when environment changes.

    Returns:
        Fresh Settings instance.
    """
    get_settings.cache_clear()
    return get_settings()
