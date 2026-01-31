"""Tests for core/claude_client.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from config.settings import CriticStrictness
from core.claude_client import ClaudeClient, ClaudeResponse, CritiqueResult


class TestClaudeResponse:
    """Tests for ClaudeResponse dataclass."""

    def test_successful_response(self):
        """Test creating a successful response."""
        response = ClaudeResponse(
            content="Test content",
            model="claude-opus-4-5-20251101",
            duration_ms=1500,
            input_tokens=100,
            output_tokens=50,
            success=True,
        )

        assert response.content == "Test content"
        assert response.model == "claude-opus-4-5-20251101"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.success is True

    def test_failed_response(self):
        """Test creating a failed response."""
        response = ClaudeResponse(
            content="",
            model="claude-opus-4-5-20251101",
            duration_ms=100,
            success=False,
            error="Rate limit exceeded",
        )

        assert not response.success
        assert response.error == "Rate limit exceeded"


class TestCritiqueResult:
    """Tests for CritiqueResult dataclass."""

    def test_passed_critique(self):
        """Test creating a passed critique result."""
        result = CritiqueResult(
            passed=True,
            feedback="Approved",
            raw_response="PASS\nContent meets all criteria.",
        )

        assert result.passed is True
        assert result.feedback == "Approved"

    def test_failed_critique(self):
        """Test creating a failed critique result."""
        result = CritiqueResult(
            passed=False,
            feedback="Missing trade-offs and business impact",
            raw_response="FAIL\nMissing trade-offs and business impact",
        )

        assert result.passed is False
        assert "trade-offs" in result.feedback


class TestClaudeClient:
    """Tests for ClaudeClient class."""

    def test_not_available_without_key(self):
        """Test client is not available without API key."""
        with patch.dict("os.environ", {}, clear=True):
            from config.settings import refresh_settings
            refresh_settings()

            client = ClaudeClient()
            assert not client.is_available()

    def test_available_with_key(self, mock_env_vars):
        """Test client is available with API key."""
        with patch("anthropic.Anthropic"):
            client = ClaudeClient()
            assert client.is_available()

    @patch("anthropic.Anthropic")
    def test_split_into_topics_success(self, mock_anthropic_class, mock_env_vars):
        """Test successful topic splitting."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='["Topic 1", "Topic 2", "Topic 3"]')]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        client = ClaudeClient()
        topics, response = client.split_into_topics("Test content about distributed systems")

        assert response.success
        assert len(topics) == 3
        assert "Topic 1" in topics

    @patch("anthropic.Anthropic")
    def test_split_into_topics_with_code_block(self, mock_anthropic_class, mock_env_vars):
        """Test topic splitting with markdown code block."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n["Topic 1", "Topic 2"]\n```')]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        client = ClaudeClient()
        topics, response = client.split_into_topics("Test content")

        assert response.success
        assert len(topics) == 2

    @patch("anthropic.Anthropic")
    def test_split_into_topics_fallback(self, mock_anthropic_class, mock_env_vars):
        """Test topic splitting fallback for non-JSON response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="- Topic 1\n- Topic 2\n- Topic 3")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        client = ClaudeClient()
        topics, response = client.split_into_topics("Test content")

        assert response.success
        assert len(topics) == 3

    @patch("anthropic.Anthropic")
    def test_critique_draft_pass(self, mock_anthropic_class, mock_env_vars):
        """Test critique that passes."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="PASS\nContent meets all requirements.")]
        mock_response.usage = MagicMock(input_tokens=200, output_tokens=20)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        client = ClaudeClient()
        result, response = client.critique_draft(
            draft="Test draft content",
            topic="Test topic",
            strictness=CriticStrictness.HIGH,
        )

        assert response.success
        assert result.passed is True
        assert result.feedback == "Approved"

    @patch("anthropic.Anthropic")
    def test_critique_draft_fail(self, mock_anthropic_class, mock_env_vars):
        """Test critique that fails."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text="FAIL\nMissing specific trade-offs.\nNo business impact mentioned."
        )]
        mock_response.usage = MagicMock(input_tokens=200, output_tokens=50)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        client = ClaudeClient()
        result, response = client.critique_draft(
            draft="Weak draft content",
            topic="Test topic",
            strictness=CriticStrictness.HIGH,
        )

        assert response.success
        assert result.passed is False
        assert "trade-offs" in result.feedback

    @patch("anthropic.Anthropic")
    def test_synthesize_guide(self, mock_anthropic_class, mock_env_vars):
        """Test guide synthesis."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="# Master Guide\n\n## Section 1\n\nContent...")]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=1000)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        client = ClaudeClient()
        sections = [
            {"topic": "Topic 1", "content": "Content 1", "low_confidence": False},
            {"topic": "Topic 2", "content": "Content 2", "low_confidence": True},
        ]
        response = client.synthesize_guide(sections)

        assert response.success
        assert "Master Guide" in response.content

    @patch("anthropic.Anthropic")
    def test_handle_api_error(self, mock_anthropic_class, mock_env_vars):
        """Test handling of API errors."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_class.return_value = mock_client

        client = ClaudeClient()
        topics, response = client.split_into_topics("Test content")

        assert not response.success
        assert "API Error" in response.error


class TestClaudeClientPrompts:
    """Tests for prompt templates."""

    def test_critique_prompts_exist_for_all_levels(self):
        """Test critique prompts exist for all strictness levels."""
        assert CriticStrictness.HIGH in ClaudeClient.CRITIQUE_PROMPTS
        assert CriticStrictness.MEDIUM in ClaudeClient.CRITIQUE_PROMPTS
        assert CriticStrictness.LOW in ClaudeClient.CRITIQUE_PROMPTS

    def test_critique_prompts_contain_placeholders(self):
        """Test critique prompts have required placeholders."""
        for level, prompt in ClaudeClient.CRITIQUE_PROMPTS.items():
            assert "{draft}" in prompt, f"Missing {{draft}} in {level}"
            assert "{topic}" in prompt, f"Missing {{topic}} in {level}"

    def test_synthesis_prompt_contains_placeholder(self):
        """Test synthesis prompt has sections placeholder."""
        assert "{sections}" in ClaudeClient.SYNTHESIS_PROMPT

    def test_high_strictness_is_more_demanding(self):
        """Test HIGH strictness prompt is more demanding than LOW."""
        high = ClaudeClient.CRITIQUE_PROMPTS[CriticStrictness.HIGH].lower()
        low = ClaudeClient.CRITIQUE_PROMPTS[CriticStrictness.LOW].lower()

        # HIGH should mention more specific criteria
        assert "ruthless" in high or "rigorous" in high
        assert "sanity check" in low or "basic" in low
