"""Tests for core/bar_raiser.py."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from config.settings import CriticStrictness
from core.bar_raiser import BarRaiser, BarRaiserResult, AttemptRecord
from core.gemini_client import GeminiClient, GeminiResponse
from core.claude_client import ClaudeClient, ClaudeResponse, CritiqueResult


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


class TestBarRaiser:
    """Tests for BarRaiser class."""

    @pytest.fixture
    def mock_clients(self):
        """Create mock Gemini and Claude clients."""
        gemini = MagicMock(spec=GeminiClient)
        claude = MagicMock(spec=ClaudeClient)
        return gemini, claude

    @pytest.fixture
    def bar_raiser(self, mock_clients):
        """Create a BarRaiser instance with mock clients."""
        gemini, claude = mock_clients
        return BarRaiser(
            gemini_client=gemini,
            claude_client=claude,
            status_callback=lambda x: None,
        )

    def test_process_topic_pass_first_attempt(self, bar_raiser, mock_clients):
        """Test topic passes on first attempt."""
        gemini, claude = mock_clients

        # Mock Gemini draft response
        gemini.generate_section_draft.return_value = GeminiResponse(
            content="Great draft content",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=True,
        )

        # Mock Claude critique - PASS
        claude.critique_draft.return_value = (
            CritiqueResult(passed=True, feedback="Approved", raw_response="PASS"),
            ClaudeResponse(
                content="PASS",
                model="claude-opus-4-5-20251101",
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

    def test_process_topic_pass_second_attempt(self, bar_raiser, mock_clients):
        """Test topic passes on second attempt."""
        gemini, claude = mock_clients

        # First draft
        gemini.generate_section_draft.return_value = GeminiResponse(
            content="First draft",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=True,
        )

        # Rewrite for second attempt
        gemini.rewrite_section.return_value = GeminiResponse(
            content="Improved draft",
            model="gemini-3-pro-preview",
            duration_ms=600,
            success=True,
        )

        # Mock critique responses - FAIL then PASS
        claude.critique_draft.side_effect = [
            (
                CritiqueResult(passed=False, feedback="Missing trade-offs", raw_response="FAIL"),
                ClaudeResponse(content="FAIL", model="claude-opus-4-5-20251101", duration_ms=300, success=True),
            ),
            (
                CritiqueResult(passed=True, feedback="Approved", raw_response="PASS"),
                ClaudeResponse(content="PASS", model="claude-opus-4-5-20251101", duration_ms=300, success=True),
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

    def test_process_topic_low_confidence_after_max_attempts(self, bar_raiser, mock_clients):
        """Test topic marked low confidence after all attempts fail."""
        gemini, claude = mock_clients

        # All drafts
        gemini.generate_section_draft.return_value = GeminiResponse(
            content="Draft content",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=True,
        )
        gemini.rewrite_section.return_value = GeminiResponse(
            content="Rewritten content",
            model="gemini-3-pro-preview",
            duration_ms=500,
            success=True,
        )

        # All critiques fail
        claude.critique_draft.return_value = (
            CritiqueResult(passed=False, feedback="Still not good enough", raw_response="FAIL"),
            ClaudeResponse(content="FAIL", model="claude-opus-4-5-20251101", duration_ms=300, success=True),
        )

        result = bar_raiser.process_topic(
            topic="Hard topic",
            context="Base knowledge",
            worker_id=1,
        )

        assert result.success is True  # Processing completed
        assert result.low_confidence is True  # But didn't pass review
        assert len(result.attempts) == 3  # max_retries + 1

    def test_strictness_relaxes_on_third_attempt(self, bar_raiser, mock_clients):
        """Test strictness relaxes to MEDIUM on third attempt."""
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
            return (
                CritiqueResult(passed=False, feedback="Fail", raw_response="FAIL"),
                ClaudeResponse(content="FAIL", model="claude-opus-4-5-20251101", duration_ms=100, success=True),
            )

        claude.critique_draft.side_effect = capture_strictness

        bar_raiser.process_topic(
            topic="Test topic",
            context="Context",
            worker_id=1,
        )

        assert strictness_levels[0] == CriticStrictness.HIGH
        assert strictness_levels[1] == CriticStrictness.HIGH
        assert strictness_levels[2] == CriticStrictness.MEDIUM

    def test_process_topics_parallel(self, bar_raiser, mock_clients):
        """Test parallel processing of multiple topics."""
        gemini, claude = mock_clients

        gemini.generate_section_draft.return_value = GeminiResponse(
            content="Draft content",
            model="gemini-3-pro-preview",
            duration_ms=100,
            success=True,
        )

        claude.critique_draft.return_value = (
            CritiqueResult(passed=True, feedback="OK", raw_response="PASS"),
            ClaudeResponse(content="PASS", model="claude-opus-4-5-20251101", duration_ms=100, success=True),
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

    def test_status_callback_called(self, mock_clients):
        """Test status callback is called during processing."""
        gemini, claude = mock_clients
        status_messages = []

        def capture_status(msg):
            status_messages.append(msg)

        bar_raiser = BarRaiser(
            gemini_client=gemini,
            claude_client=claude,
            status_callback=capture_status,
        )

        gemini.generate_section_draft.return_value = GeminiResponse(
            content="Draft", model="gemini-3-pro-preview", duration_ms=100, success=True
        )
        claude.critique_draft.return_value = (
            CritiqueResult(passed=True, feedback="OK", raw_response="PASS"),
            ClaudeResponse(content="PASS", model="claude-opus-4-5-20251101", duration_ms=100, success=True),
        )

        bar_raiser.process_topic("Topic", "Context", 1)

        assert len(status_messages) > 0
        assert any("Worker 1" in msg for msg in status_messages)
