"""Tests for utils/logging_utils.py."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.logging_utils import RequestLogger, configure_logging


class TestRequestLogger:
    """Tests for RequestLogger class."""

    @pytest.fixture
    def logger(self, temp_dir):
        """Create a RequestLogger with temp directory."""
        log_path = temp_dir / "request_history.json"
        return RequestLogger(log_path=str(log_path))

    def test_creates_log_file(self, temp_dir):
        """Test that logger creates log file on init."""
        log_path = temp_dir / "test_log.json"
        logger = RequestLogger(log_path=str(log_path))

        assert log_path.exists()

        with open(log_path, "r") as f:
            data = json.load(f)
        assert "sessions" in data

    def test_log_session(self, logger, sample_pipeline_result, temp_dir):
        """Test logging a pipeline session."""
        logger.log_session(sample_pipeline_result)

        # Read log file
        log_path = temp_dir / "request_history.json"
        with open(log_path, "r") as f:
            data = json.load(f)

        assert len(data["sessions"]) == 1
        session = data["sessions"][0]
        assert session["id"] == "test-session-123"
        assert session["topic"] == "Test Topic"
        assert session["success"] is True

    def test_log_multiple_sessions(self, logger, sample_pipeline_result, temp_dir):
        """Test logging multiple sessions."""
        logger.log_session(sample_pipeline_result)
        logger.log_session(sample_pipeline_result)
        logger.log_session(sample_pipeline_result)

        log_path = temp_dir / "request_history.json"
        with open(log_path, "r") as f:
            data = json.load(f)

        assert len(data["sessions"]) == 3

    def test_get_recent_sessions(self, logger, sample_pipeline_result):
        """Test getting recent sessions."""
        # Log some sessions
        for i in range(5):
            logger.log_session(sample_pipeline_result)

        recent = logger.get_recent_sessions(limit=3)

        assert len(recent) == 3

    def test_get_recent_sessions_empty(self, logger):
        """Test getting recent sessions when none exist."""
        recent = logger.get_recent_sessions()
        assert recent == []

    def test_get_session_by_id(self, logger, sample_pipeline_result):
        """Test retrieving a specific session."""
        logger.log_session(sample_pipeline_result)

        session = logger.get_session("test-session-123")

        assert session is not None
        assert session["id"] == "test-session-123"

    def test_get_session_not_found(self, logger):
        """Test retrieving non-existent session."""
        session = logger.get_session("nonexistent-id")
        assert session is None

    def test_session_contains_steps(self, logger, sample_pipeline_result, temp_dir):
        """Test that logged session contains step details."""
        logger.log_session(sample_pipeline_result)

        log_path = temp_dir / "request_history.json"
        with open(log_path, "r") as f:
            data = json.load(f)

        session = data["sessions"][0]
        assert "steps" in session
        assert len(session["steps"]) > 0
        assert session["steps"][0]["name"] == "base_knowledge"

    def test_session_contains_deep_dives(self, logger, sample_pipeline_result, temp_dir):
        """Test that logged session contains deep dive details."""
        logger.log_session(sample_pipeline_result)

        log_path = temp_dir / "request_history.json"
        with open(log_path, "r") as f:
            data = json.load(f)

        session = data["sessions"][0]
        assert "deep_dives" in session
        assert len(session["deep_dives"]) > 0
        assert "attempts" in session["deep_dives"][0]

    def test_deep_dive_contains_attempts(self, logger, sample_pipeline_result, temp_dir):
        """Test that deep dives contain attempt details."""
        logger.log_session(sample_pipeline_result)

        log_path = temp_dir / "request_history.json"
        with open(log_path, "r") as f:
            data = json.load(f)

        deep_dive = data["sessions"][0]["deep_dives"][0]
        assert len(deep_dive["attempts"]) > 0

        attempt = deep_dive["attempts"][0]
        assert "attempt_number" in attempt
        assert "strictness" in attempt
        assert "critique_passed" in attempt
        assert "draft_preview" in attempt


class TestLogRotation:
    """Tests for log rotation functionality."""

    def test_rotates_old_logs(self, temp_dir):
        """Test that old logs are rotated."""
        log_path = temp_dir / "request_history.json"

        # Create log with old entries
        old_date = (datetime.now() - timedelta(days=35)).isoformat()
        recent_date = datetime.now().isoformat()

        data = {
            "sessions": [
                {"id": "old-session", "timestamp": old_date, "topic": "Old"},
                {"id": "new-session", "timestamp": recent_date, "topic": "New"},
            ]
        }

        with open(log_path, "w") as f:
            json.dump(data, f)

        # Create logger (triggers rotation on init)
        with patch.dict(os.environ, {"LOG_RETENTION_DAYS": "30"}):
            from config.settings import refresh_settings
            refresh_settings()
            logger = RequestLogger(log_path=str(log_path))

        # Read and verify
        with open(log_path, "r") as f:
            data = json.load(f)

        # Old session should be removed
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "new-session"

    def test_keeps_recent_logs(self, temp_dir):
        """Test that recent logs are kept during rotation."""
        log_path = temp_dir / "request_history.json"

        recent_date = datetime.now().isoformat()

        data = {
            "sessions": [
                {"id": "session-1", "timestamp": recent_date, "topic": "Recent 1"},
                {"id": "session-2", "timestamp": recent_date, "topic": "Recent 2"},
            ]
        }

        with open(log_path, "w") as f:
            json.dump(data, f)

        logger = RequestLogger(log_path=str(log_path))

        with open(log_path, "r") as f:
            data = json.load(f)

        assert len(data["sessions"]) == 2


class TestConfigureLogging:
    """Tests for logging configuration."""

    def test_configure_with_debug(self):
        """Test configuring with DEBUG level."""
        configure_logging("DEBUG")
        # Just ensure it doesn't raise

    def test_configure_with_info(self):
        """Test configuring with INFO level."""
        configure_logging("INFO")

    def test_configure_with_invalid_level(self):
        """Test configuring with invalid level defaults gracefully."""
        configure_logging("INVALID_LEVEL")
        # Should not raise, defaults to INFO
