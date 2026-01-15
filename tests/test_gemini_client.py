"""Tests for core/gemini_client.py."""

from unittest.mock import MagicMock, patch

import pytest

from core.gemini_client import GeminiClient, GeminiResponse


class TestGeminiResponse:
    """Tests for GeminiResponse dataclass."""

    def test_successful_response(self):
        """Test creating a successful response."""
        response = GeminiResponse(
            content="Test content",
            model="gemini-3-pro-preview",
            duration_ms=1000,
            success=True,
        )

        assert response.content == "Test content"
        assert response.model == "gemini-3-pro-preview"
        assert response.duration_ms == 1000
        assert response.success is True
        assert response.error is None

    def test_failed_response(self):
        """Test creating a failed response."""
        response = GeminiResponse(
            content="",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=False,
            error="API timeout",
        )

        assert response.content == ""
        assert response.success is False
        assert response.error == "API timeout"


class TestGeminiClient:
    """Tests for GeminiClient class."""

    def test_not_available_without_key(self):
        """Test client is not available without API key."""
        with patch.dict("os.environ", {}, clear=True):
            from config.settings import refresh_settings
            refresh_settings()

            client = GeminiClient()
            assert not client.is_available()

    def test_available_with_key(self, mock_env_vars):
        """Test client is available with API key."""
        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel"):
                client = GeminiClient()
                assert client.is_available()

    def test_generate_base_knowledge_not_configured(self):
        """Test generate returns error when not configured."""
        with patch.dict("os.environ", {}, clear=True):
            from config.settings import refresh_settings
            refresh_settings()

            client = GeminiClient()
            response = client.generate_base_knowledge("Test topic")

            assert not response.success
            assert "not configured" in response.error

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_generate_base_knowledge_success(
        self, mock_configure, mock_model_class, mock_env_vars
    ):
        """Test successful base knowledge generation."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(
            text="Generated base knowledge content"
        )
        mock_model_class.return_value = mock_model

        client = GeminiClient()
        response = client.generate_base_knowledge("Distributed systems")

        assert response.success
        assert response.content == "Generated base knowledge content"
        assert response.duration_ms >= 0

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_generate_section_draft(
        self, mock_configure, mock_model_class, mock_env_vars
    ):
        """Test section draft generation."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(
            text="Section draft content"
        )
        mock_model_class.return_value = mock_model

        client = GeminiClient()
        response = client.generate_section_draft(
            topic="Consensus algorithms",
            context="Base knowledge about distributed systems",
            feedback="",
        )

        assert response.success
        assert response.content == "Section draft content"

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_rewrite_section(
        self, mock_configure, mock_model_class, mock_env_vars
    ):
        """Test section rewrite with critique feedback."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(
            text="Improved section content"
        )
        mock_model_class.return_value = mock_model

        client = GeminiClient()
        response = client.rewrite_section(
            topic="Consensus algorithms",
            previous_draft="Original draft",
            critique="Missing trade-offs",
            strictness="high",
        )

        assert response.success
        assert response.content == "Improved section content"

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_handle_api_error(
        self, mock_configure, mock_model_class, mock_env_vars
    ):
        """Test handling of API errors."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model

        client = GeminiClient()
        response = client.generate_base_knowledge("Test topic")

        assert not response.success
        assert "API Error" in response.error

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_handle_empty_response(
        self, mock_configure, mock_model_class, mock_env_vars
    ):
        """Test handling of empty response."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="")
        mock_model_class.return_value = mock_model

        client = GeminiClient()
        response = client.generate_base_knowledge("Test topic")

        assert response.success
        assert response.content == ""


class TestGeminiClientPrompts:
    """Tests for prompt templates."""

    def test_base_knowledge_prompt_contains_topic(self):
        """Test base knowledge prompt includes topic placeholder."""
        assert "{topic}" in GeminiClient.BASE_KNOWLEDGE_PROMPT

    def test_section_draft_prompt_contains_required_placeholders(self):
        """Test section draft prompt includes all placeholders."""
        prompt = GeminiClient.SECTION_DRAFT_PROMPT
        assert "{topic}" in prompt
        assert "{context}" in prompt
        assert "{feedback}" in prompt

    def test_rewrite_prompt_contains_required_placeholders(self):
        """Test rewrite prompt includes all placeholders."""
        prompt = GeminiClient.SECTION_REWRITE_PROMPT
        assert "{topic}" in prompt
        assert "{previous_draft}" in prompt
        assert "{critique}" in prompt
        assert "{strictness}" in prompt
