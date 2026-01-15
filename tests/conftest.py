"""Pytest configuration and fixtures for Professor Gemini tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings, refresh_settings


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings cache before each test."""
    refresh_settings()
    yield
    refresh_settings()


@pytest.fixture
def mock_env_vars():
    """Provide mock environment variables for testing."""
    env_vars = {
        "GEMINI_API_KEY": "test_gemini_key_12345",
        "ANTHROPIC_API_KEY": "test_anthropic_key_12345",
        "GEMINI_MODEL": "gemini-3-pro-preview",
        "CLAUDE_MODEL": "claude-opus-4-5-20251101",
        "MAX_WORKERS": "5",
        "MAX_RETRIES": "2",
        "API_TIMEOUT": "60",
        "LOG_LEVEL": "DEBUG",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        refresh_settings()
        yield env_vars


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_cyrus_root(temp_dir):
    """Create a mock Cyrus project structure."""
    cyrus_root = temp_dir / "Cyrus"
    cyrus_root.mkdir()
    (cyrus_root / "gemini-responses").mkdir()
    return cyrus_root


@pytest.fixture
def sample_pipeline_result():
    """Provide a sample pipeline result for testing."""
    from core.pipeline import PipelineResult, PipelineStep
    from core.bar_raiser import BarRaiserResult, AttemptRecord
    from config.settings import CriticStrictness
    from datetime import datetime

    return PipelineResult(
        session_id="test-session-123",
        topic="Test Topic",
        master_guide="# Test Guide\n\nThis is a test guide.",
        low_confidence_sections=1,
        total_sections=3,
        steps=[
            PipelineStep(
                name="base_knowledge",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                duration_ms=1000,
                success=True,
                metadata={"model": "gemini-3-pro-preview"},
            ),
        ],
        deep_dive_results=[
            BarRaiserResult(
                topic="Sub-topic 1",
                final_content="Content for topic 1",
                low_confidence=False,
                attempts=[
                    AttemptRecord(
                        attempt_number=1,
                        strictness=CriticStrictness.HIGH,
                        draft="Draft content",
                        critique_passed=True,
                        critique_feedback="Approved",
                        draft_duration_ms=500,
                        critique_duration_ms=300,
                    ),
                ],
                total_duration_ms=800,
                success=True,
            ),
        ],
        total_duration_ms=5000,
        success=True,
    )


@pytest.fixture
def mock_gemini_response():
    """Provide a mock Gemini API response."""
    mock = MagicMock()
    mock.text = "This is a test response from Gemini."
    return mock


@pytest.fixture
def mock_claude_response():
    """Provide a mock Claude API response."""
    from anthropic.types import Message, TextBlock, Usage

    mock = MagicMock(spec=Message)
    mock.content = [MagicMock(spec=TextBlock, text="PASS\nApproved content.")]
    mock.usage = MagicMock(spec=Usage, input_tokens=100, output_tokens=50)
    return mock
