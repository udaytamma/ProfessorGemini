"""Tests for core/perplexity_client.py."""

import os
from unittest.mock import MagicMock, patch

import pytest

from core.perplexity_client import PerplexityClient, PerplexityResponse


class TestPerplexityResponse:
    """Tests for PerplexityResponse dataclass."""

    def test_successful_response(self):
        """Test creating a successful response."""
        response = PerplexityResponse(
            content="Paris is the capital of France.",
            model="sonar-pro",
            duration_ms=2500,
            success=True,
        )

        assert response.content == "Paris is the capital of France."
        assert response.model == "sonar-pro"
        assert response.duration_ms == 2500
        assert response.success is True
        assert response.error is None

    def test_failed_response(self):
        """Test creating a failed response."""
        response = PerplexityResponse(
            content="",
            model="sonar-pro",
            duration_ms=100,
            success=False,
            error="Rate limit exceeded",
        )

        assert not response.success
        assert response.error == "Rate limit exceeded"
        assert response.content == ""

    def test_default_values(self):
        """Test default values for optional fields."""
        response = PerplexityResponse(
            content="Test",
            model="sonar-pro",
            duration_ms=1000,
        )

        assert response.success is True
        assert response.error is None


class TestPerplexityClient:
    """Tests for PerplexityClient class."""

    def test_not_available_without_key(self):
        """Test client is not available without API key."""
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": ""}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            client = PerplexityClient()
            assert not client.is_available()

    @patch("core.perplexity_client.OpenAI")
    def test_available_with_key(self, mock_openai_class, mock_perplexity_env):
        """Test client is available with API key."""
        client = PerplexityClient()
        assert client.is_available()

    def test_base_url_is_correct(self):
        """Test Perplexity base URL is set correctly."""
        assert PerplexityClient.BASE_URL == "https://api.perplexity.ai"

    @patch("core.perplexity_client.OpenAI")
    def test_search_success(self, mock_openai_class, mock_perplexity_env):
        """Test successful search query."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Paris is the capital of France."))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = PerplexityClient()
        result = client.search("What is the capital of France?")

        assert result.success
        assert "Paris" in result.content
        assert result.model == "sonar-pro"
        assert result.duration_ms >= 0

    @patch("core.perplexity_client.OpenAI")
    def test_search_empty_response(self, mock_openai_class, mock_perplexity_env):
        """Test handling empty response from API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = PerplexityClient()
        result = client.search("Test query")

        assert result.success
        assert result.content == ""

    @patch("core.perplexity_client.OpenAI")
    def test_search_none_content(self, mock_openai_class, mock_perplexity_env):
        """Test handling None content in response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = PerplexityClient()
        result = client.search("Test query")

        assert result.success
        assert result.content == ""

    @patch("core.perplexity_client.OpenAI")
    def test_search_api_error(self, mock_openai_class, mock_perplexity_env):
        """Test handling of API errors."""
        # Use a mock that raises APIError-like exception
        mock_client = MagicMock()
        mock_error = MagicMock()
        mock_error.__str__ = lambda self: "Internal server error"
        mock_client.chat.completions.create.side_effect = Exception("Internal server error")
        mock_openai_class.return_value = mock_client

        client = PerplexityClient()
        result = client.search("Test query")

        assert not result.success
        assert result.error is not None
        assert "Internal server error" in result.error

    @patch("core.perplexity_client.OpenAI")
    def test_search_timeout_error(self, mock_openai_class, mock_perplexity_env):
        """Test handling of timeout errors."""
        from openai import APITimeoutError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )
        mock_openai_class.return_value = mock_client

        client = PerplexityClient()
        result = client.search("Test query")

        assert not result.success
        assert "timeout" in result.error.lower()

    @patch("core.perplexity_client.OpenAI")
    def test_search_generic_exception(self, mock_openai_class, mock_perplexity_env):
        """Test handling of unexpected exceptions."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Unexpected error")
        mock_openai_class.return_value = mock_client

        client = PerplexityClient()
        result = client.search("Test query")

        assert not result.success
        assert "Unexpected error" in result.error

    def test_search_without_configuration(self):
        """Test search returns error when client not configured."""
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": ""}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            client = PerplexityClient()
            result = client.search("Test query")

            assert not result.success
            assert "not configured" in result.error.lower()


class TestPerplexityClientConfiguration:
    """Tests for Perplexity client configuration."""

    @patch("core.perplexity_client.OpenAI")
    def test_uses_correct_model(self, mock_openai_class, mock_perplexity_env):
        """Test client uses sonar-pro model by default."""
        from config.settings import get_settings
        settings = get_settings()
        assert settings.perplexity_model == "sonar-pro"

    @patch("core.perplexity_client.OpenAI")
    def test_system_prompt_is_set(self, mock_openai_class, mock_perplexity_env):
        """Test search includes appropriate system prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        client = PerplexityClient()
        client.search("Test query")

        # Verify the call was made with messages
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "research assistant" in messages[0]["content"].lower()


class TestPerplexityClientIntegration:
    """Integration tests that hit the actual API."""

    @pytest.mark.integration
    def test_real_api_call(self):
        """Test actual API call to Perplexity."""
        from config.settings import refresh_settings
        refresh_settings()

        client = PerplexityClient()
        if not client.is_available():
            pytest.skip("Perplexity client not configured")

        result = client.search("What is 2 + 2?")

        assert result.success
        assert result.content
        assert "4" in result.content or "four" in result.content.lower()
        assert result.duration_ms > 0


# Fixture for Perplexity environment
@pytest.fixture
def mock_perplexity_env():
    """Provide mock Perplexity environment variables."""
    env_vars = {
        "PERPLEXITY_API_KEY": "test_perplexity_key_12345",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        from config.settings import refresh_settings
        refresh_settings()
        yield env_vars
