"""Tests for core/gemini_client.py."""

import os
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
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            # Mock the genai.Client to prevent real API calls
            with patch("google.genai.Client"):
                client = GeminiClient()
                assert not client.is_available()

    def test_available_with_key(self, mock_env_vars):
        """Test client is available with API key."""
        with patch("google.genai.Client") as mock_client:
            mock_client.return_value = MagicMock()
            client = GeminiClient()
            assert client.is_available()

    def test_generate_base_knowledge_not_configured(self):
        """Test generate returns error when not configured."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            with patch("google.genai.Client"):
                client = GeminiClient()
                response = client.generate_base_knowledge("Test topic")

                assert not response.success
                assert "not configured" in response.error

    def test_generate_base_knowledge_success(self, mock_env_vars):
        """Test successful base knowledge generation."""
        with patch("google.genai.Client") as mock_client_class:
            # Setup mock
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Generated base knowledge content"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            response = client.generate_base_knowledge("Distributed systems")

            assert response.success
            assert response.content == "Generated base knowledge content"
            assert response.duration_ms >= 0

    def test_generate_section_draft(self, mock_env_vars):
        """Test section draft generation."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Section draft content"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            response = client.generate_section_draft(
                topic="Consensus algorithms",
                context="Base knowledge about distributed systems",
                feedback="",
            )

            assert response.success
            assert response.content == "Section draft content"

    def test_rewrite_section(self, mock_env_vars):
        """Test section rewrite with critique feedback."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Improved section content"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            response = client.rewrite_section(
                topic="Consensus algorithms",
                previous_draft="Original draft",
                critique="Missing trade-offs",
                strictness="high",
            )

            assert response.success
            assert response.content == "Improved section content"

    def test_handle_api_error(self, mock_env_vars):
        """Test handling of API errors."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("API Error")
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            response = client.generate_base_knowledge("Test topic")

            assert not response.success
            assert "API Error" in response.error

    def test_handle_empty_response(self, mock_env_vars):
        """Test handling of empty response."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = ""
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

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


class TestGeminiOnlyMethods:
    """Tests for Gemini-only mode methods."""

    def test_split_into_topics_success(self, mock_env_vars):
        """Test splitting content into topics."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = '["Topic 1", "Topic 2", "Topic 3"]'
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            topics, response = client.split_into_topics("Base content")

            assert response.success
            assert len(topics) == 3
            assert "Topic 1" in topics

    def test_split_into_topics_with_markdown(self, mock_env_vars):
        """Test splitting topics handles markdown code blocks."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = '```json\n["Topic A", "Topic B"]\n```'
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            topics, response = client.split_into_topics("Content")

            assert len(topics) == 2

    def test_split_into_topics_fallback(self, mock_env_vars):
        """Test fallback when JSON parsing fails."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "- Topic One\n- Topic Two\n- Topic Three"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            topics, response = client.split_into_topics("Content")

            assert len(topics) == 3

    def test_critique_draft_pass(self, mock_env_vars):
        """Test critique that passes."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "PASS\nContent looks good."
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            result, response = client.critique_draft("Draft", "Topic", "high")

            assert result["passed"] is True
            assert "Approved" in result["feedback"]

    def test_critique_draft_fail(self, mock_env_vars):
        """Test critique that fails."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "FAIL\nMissing trade-offs\nNeeds more depth"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            result, response = client.critique_draft("Draft", "Topic", "high")

            assert result["passed"] is False
            assert "Missing trade-offs" in result["feedback"]

    def test_synthesize_guide(self, mock_env_vars):
        """Test synthesizing multiple sections into guide."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "# Master Guide\n\n## Section 1\nContent..."
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient()
            sections = [
                {"topic": "Topic 1", "content": "Content 1", "low_confidence": False},
                {"topic": "Topic 2", "content": "Content 2", "low_confidence": True},
            ]
            response = client.synthesize_guide(sections)

            assert response.success
            assert "Master Guide" in response.content
