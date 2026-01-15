"""Tests for config/settings.py."""

import os
from unittest.mock import patch

import pytest

from config.settings import (
    Settings,
    CriticStrictness,
    get_settings,
    refresh_settings,
)


class TestCriticStrictness:
    """Tests for CriticStrictness enum."""

    def test_strictness_values(self):
        """Test strictness enum has expected values."""
        assert CriticStrictness.LOW.value == "low"
        assert CriticStrictness.MEDIUM.value == "medium"
        assert CriticStrictness.HIGH.value == "high"

    def test_strictness_from_string(self):
        """Test creating strictness from string."""
        assert CriticStrictness("low") == CriticStrictness.LOW
        assert CriticStrictness("medium") == CriticStrictness.MEDIUM
        assert CriticStrictness("high") == CriticStrictness.HIGH


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Test default setting values."""
        settings = Settings()

        assert settings.gemini_model == "gemini-3-pro-preview"
        assert settings.claude_model == "claude-opus-4-5-20251101"
        assert settings.max_workers == 10
        assert settings.max_retries == 2
        assert settings.api_timeout == 120
        assert settings.log_retention_days == 30
        assert settings.app_port == 8502
        assert settings.default_strictness == CriticStrictness.HIGH

    def test_gemini_not_configured_by_default(self):
        """Test Gemini is not configured without API key."""
        settings = Settings()
        assert not settings.is_gemini_configured()

    def test_claude_not_configured_by_default(self):
        """Test Claude is not configured without API key."""
        settings = Settings()
        assert not settings.is_claude_configured()

    def test_fully_configured_requires_both(self):
        """Test fully configured requires both API keys."""
        settings = Settings()
        assert not settings.is_fully_configured()

    def test_with_env_vars(self, mock_env_vars):
        """Test settings load from environment variables."""
        settings = get_settings()

        assert settings.gemini_api_key == "test_gemini_key_12345"
        assert settings.anthropic_api_key == "test_anthropic_key_12345"
        assert settings.is_gemini_configured()
        assert settings.is_claude_configured()
        assert settings.is_fully_configured()

    def test_max_workers_bounds(self, mock_env_vars):
        """Test max workers loaded from env."""
        settings = get_settings()
        assert settings.max_workers == 5

    def test_api_timeout_from_env(self, mock_env_vars):
        """Test API timeout loaded from env."""
        settings = get_settings()
        assert settings.api_timeout == 60

    def test_get_gemini_responses_path(self):
        """Test gemini responses path generation."""
        settings = Settings(cyrus_root_path="/test/path")
        path = settings.get_gemini_responses_path()
        assert path == "/test/path/gemini-responses"

    def test_whitespace_stripping(self):
        """Test API keys have whitespace stripped."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "  key_with_spaces  "}):
            refresh_settings()
            settings = get_settings()
            assert settings.gemini_api_key == "key_with_spaces"


class TestGetSettings:
    """Tests for get_settings function."""

    def test_caching(self, mock_env_vars):
        """Test settings are cached."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_refresh_clears_cache(self, mock_env_vars):
        """Test refresh_settings clears the cache."""
        settings1 = get_settings()
        refresh_settings()
        settings2 = get_settings()
        # Objects should be different instances after refresh
        assert settings1 is not settings2
