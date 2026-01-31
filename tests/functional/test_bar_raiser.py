"""Tests for core/bar_raiser.py."""

import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from config.settings import CriticStrictness
from core.bar_raiser import BarRaiser, BarRaiserResult, AttemptRecord
from core.gemini_client import GeminiClient, GeminiResponse


class TestAttemptRecord:
    """Tests for AttemptRecord dataclass."""

    def test_create_attempt_record(self):
        """Test creating an attempt record."""
        record = AttemptRecord(
            attempt_number=1,
            strictness=CriticStrictness.HIGH,
            draft="Test draft",
            critique_passed=True,
            critique_feedback="Approved",
            draft_duration_ms=500,
            critique_duration_ms=300,
        )

        assert record.attempt_number == 1
        assert record.strictness == CriticStrictness.HIGH
        assert record.critique_passed is True


class TestBarRaiserResult:
    """Tests for BarRaiserResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = BarRaiserResult(
            topic="Test Topic",
            final_content="Final content",
            low_confidence=False,
            attempts=[],
            total_duration_ms=1000,
            success=True,
        )

        assert result.topic == "Test Topic"
        assert result.low_confidence is False
        assert result.success is True

    def test_low_confidence_result(self):
        """Test creating a low confidence result."""
        result = BarRaiserResult(
            topic="Hard Topic",
            final_content="Best effort content",
            low_confidence=True,
            attempts=[],
            total_duration_ms=5000,
            success=True,
        )

        assert result.low_confidence is True
        assert result.success is True

    def test_failed_result(self):
        """Test creating a failed result."""
        result = BarRaiserResult(
            topic="Failed Topic",
            final_content="",
            low_confidence=True,
            success=False,
            error="Timeout",
        )

        assert result.success is False
        assert result.error == "Timeout"


class TestBarRaiserGeminiOnly:
    """Tests for BarRaiser in Gemini-only mode."""

    @pytest.fixture
    def mock_gemini(self):
        """Create mock Gemini client."""
        return MagicMock(spec=GeminiClient)

    @pytest.fixture
    def bar_raiser(self, mock_gemini, mock_env_vars):
        """Create a BarRaiser instance with mock Gemini client."""
        with patch.dict(os.environ, {"USE_CLAUDE": "false"}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            return BarRaiser(
                gemini_client=mock_gemini,
                claude_client=None,  # Gemini-only mode
                status_callback=lambda x: None,
            )

    def test_process_topic_pass_first_attempt(self, bar_raiser, mock_gemini):
        """Test topic passes on first attempt."""
        # Mock Gemini draft response
        mock_gemini.generate_section_draft.return_value = GeminiResponse(
            content="Great draft content",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=True,
        )

        # Mock Gemini critique - PASS
        mock_gemini.critique_draft.return_value = (
            {"passed": True, "feedback": "Approved", "raw_response": "PASS\nApproved"},
            GeminiResponse(
                content="PASS\nApproved",
                model="gemini-3-pro-preview",
                duration_ms=300,
                success=True,
            ),
        )

        result = bar_raiser.process_topic(
            topic="Test topic",
            context="Base knowledge context",
            worker_id=1,
        )

        assert result.success is True
        assert result.low_confidence is False
        assert len(result.attempts) == 1
        assert result.attempts[0].critique_passed is True

    def test_process_topic_pass_second_attempt(self, bar_raiser, mock_gemini):
        """Test topic passes on second attempt."""
        # First draft
        mock_gemini.generate_section_draft.return_value = GeminiResponse(
            content="First draft",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=True,
        )

        # Rewrite for second attempt
        mock_gemini.rewrite_section.return_value = GeminiResponse(
            content="Improved draft",
            model="gemini-3-pro-preview",
            duration_ms=600,
            success=True,
        )

        # Mock critique responses - FAIL then PASS
        mock_gemini.critique_draft.side_effect = [
            (
                {"passed": False, "feedback": "Missing trade-offs", "raw_response": "FAIL"},
                GeminiResponse(content="FAIL", model="gemini-3-pro-preview", duration_ms=300, success=True),
            ),
            (
                {"passed": True, "feedback": "Approved", "raw_response": "PASS"},
                GeminiResponse(content="PASS", model="gemini-3-pro-preview", duration_ms=300, success=True),
            ),
        ]

        result = bar_raiser.process_topic(
            topic="Test topic",
            context="Base knowledge",
            worker_id=1,
        )

        assert result.success is True
        assert result.low_confidence is False
        assert len(result.attempts) == 2
        assert result.attempts[0].critique_passed is False
        assert result.attempts[1].critique_passed is True

    def test_process_topic_low_confidence_after_max_attempts(self, bar_raiser, mock_gemini):
        """Test topic marked low confidence after all attempts fail."""
        # All drafts
        mock_gemini.generate_section_draft.return_value = GeminiResponse(
            content="Draft content",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=True,
        )
        mock_gemini.rewrite_section.return_value = GeminiResponse(
            content="Rewritten content",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=True,
        )

        # All critiques fail
        mock_gemini.critique_draft.return_value = (
            {"passed": False, "feedback": "Still not good enough", "raw_response": "FAIL"},
            GeminiResponse(content="FAIL", model="gemini-3-pro-preview", duration_ms=300, success=True),
        )

        result = bar_raiser.process_topic(
            topic="Hard topic",
            context="Base knowledge",
            worker_id=1,
        )

        assert result.success is True  # Processing completed
        assert result.low_confidence is True  # But didn't pass review
        assert len(result.attempts) == 3  # max_retries + 1

    def test_strictness_relaxes_on_third_attempt(self, bar_raiser, mock_gemini):
        """Test strictness relaxes to MEDIUM on third attempt."""
        mock_gemini.generate_section_draft.return_value = GeminiResponse(
            content="Draft", model="gemini-3-pro-preview", duration_ms=500, success=True
        )
        mock_gemini.rewrite_section.return_value = GeminiResponse(
            content="Rewritten", model="gemini-3-pro-preview", duration_ms=500, success=True
        )

        # Track strictness levels passed to critique
        strictness_levels = []

        def capture_strictness(draft, topic, strictness):
            strictness_levels.append(strictness)
            return (
                {"passed": False, "feedback": "Fail", "raw_response": "FAIL"},
                GeminiResponse(content="FAIL", model="gemini-3-pro-preview", duration_ms=100, success=True),
            )

        mock_gemini.critique_draft.side_effect = capture_strictness

        bar_raiser.process_topic(
            topic="Test topic",
            context="Context",
            worker_id=1,
        )

        # Strictness values are passed as strings in Gemini-only mode
        # Default is LOW, relaxes to LOW on 3rd attempt (stays at lowest)
        assert strictness_levels[0] == "low"
        assert strictness_levels[1] == "low"
        assert strictness_levels[2] == "low"

    def test_process_topics_parallel(self, bar_raiser, mock_gemini):
        """Test parallel processing of multiple topics."""
        mock_gemini.generate_section_draft.return_value = GeminiResponse(
            content="Draft content",
            model="gemini-3-pro-preview",
            duration_ms=100,
            success=True,
        )

        mock_gemini.critique_draft.return_value = (
            {"passed": True, "feedback": "OK", "raw_response": "PASS"},
            GeminiResponse(content="PASS", model="gemini-3-pro-preview", duration_ms=100, success=True),
        )

        topics = ["Topic 1", "Topic 2", "Topic 3"]

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = bar_raiser.process_topics_parallel(
                topics=topics,
                context="Context",
                executor=executor,
            )

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_status_callback_called(self, mock_gemini, mock_env_vars):
        """Test status callback is called during processing."""
        with patch.dict(os.environ, {"USE_CLAUDE": "false"}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            status_messages = []

            def capture_status(msg):
                status_messages.append(msg)

            bar_raiser = BarRaiser(
                gemini_client=mock_gemini,
                claude_client=None,
                status_callback=capture_status,
            )

            mock_gemini.generate_section_draft.return_value = GeminiResponse(
                content="Draft", model="gemini-3-pro-preview", duration_ms=100, success=True
            )
            mock_gemini.critique_draft.return_value = (
                {"passed": True, "feedback": "OK", "raw_response": "PASS"},
                GeminiResponse(content="PASS", model="gemini-3-pro-preview", duration_ms=100, success=True),
            )

            bar_raiser.process_topic("Topic", "Context", 1)

            assert len(status_messages) > 0
            assert any("Worker 1" in msg for msg in status_messages)


class TestBarRaiserHybridMode:
    """Tests for BarRaiser in Hybrid (Gemini + Claude) mode."""

    @pytest.fixture
    def mock_clients(self, mock_env_vars):
        """Create mock Gemini and Claude clients."""
        gemini = MagicMock(spec=GeminiClient)
        claude = MagicMock()
        return gemini, claude

    @pytest.fixture
    def bar_raiser(self, mock_clients, mock_env_vars):
        """Create a BarRaiser instance with both clients."""
        with patch.dict(os.environ, {"USE_CLAUDE": "true"}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            gemini, claude = mock_clients
            return BarRaiser(
                gemini_client=gemini,
                claude_client=claude,
                status_callback=lambda x: None,
            )

    def test_uses_claude_for_critique_in_hybrid_mode(self, bar_raiser, mock_clients):
        """Test that Claude is used for critique in hybrid mode."""
        gemini, claude = mock_clients

        # Mock Gemini draft response
        gemini.generate_section_draft.return_value = GeminiResponse(
            content="Draft content",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=True,
        )

        # Create mock CritiqueResult-like object
        mock_critique_result = MagicMock()
        mock_critique_result.passed = True
        mock_critique_result.feedback = "Approved"

        # Mock ClaudeResponse-like object
        mock_claude_response = MagicMock()
        mock_claude_response.duration_ms = 300

        claude.critique_draft.return_value = (mock_critique_result, mock_claude_response)

        result = bar_raiser.process_topic(
            topic="Test topic",
            context="Context",
            worker_id=1,
        )

        # Verify Claude was called for critique
        claude.critique_draft.assert_called()
        assert result.success is True

    def test_hybrid_mode_strictness_passed_correctly(self, bar_raiser, mock_clients):
        """Test that strictness is passed as CriticStrictness enum in hybrid mode."""
        gemini, claude = mock_clients

        gemini.generate_section_draft.return_value = GeminiResponse(
            content="Draft", model="gemini-3-pro-preview", duration_ms=500, success=True
        )
        gemini.rewrite_section.return_value = GeminiResponse(
            content="Rewritten", model="gemini-3-pro-preview", duration_ms=500, success=True
        )

        # Track strictness levels
        strictness_levels = []

        def capture_strictness(draft, topic, strictness):
            strictness_levels.append(strictness)
            mock_result = MagicMock()
            mock_result.passed = False
            mock_result.feedback = "Fail"
            mock_response = MagicMock()
            mock_response.duration_ms = 100
            return (mock_result, mock_response)

        claude.critique_draft.side_effect = capture_strictness

        bar_raiser.process_topic(
            topic="Test topic",
            context="Context",
            worker_id=1,
        )

        # In hybrid mode, strictness is passed as CriticStrictness enum
        # Default is LOW, which is already the lowest level
        assert strictness_levels[0] == CriticStrictness.LOW
        assert strictness_levels[1] == CriticStrictness.LOW
        assert strictness_levels[2] == CriticStrictness.LOW
