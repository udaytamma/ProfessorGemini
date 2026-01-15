"""Request history logging utilities.

Provides persistent logging of all pipeline executions with automatic
log rotation to manage disk space.
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from config.settings import get_settings
from core.pipeline import PipelineResult
from core.bar_raiser import BarRaiserResult, AttemptRecord


logger = logging.getLogger(__name__)


class RequestLogger:
    """Manages request history logging with JSON persistence.

    Logs all pipeline executions including drafts, critiques, and retries
    to enable full traceability. Auto-rotates logs older than configured days.
    """

    def __init__(self, log_path: Optional[str] = None) -> None:
        """Initialize request logger.

        Args:
            log_path: Optional custom path for log file.
        """
        self._settings = get_settings()

        if log_path:
            self._log_path = Path(log_path)
        else:
            # Default to project root
            self._log_path = Path(__file__).parent.parent / "request_history.json"

        self._lock = Lock()
        self._ensure_log_file()
        self._rotate_old_logs()

    def _ensure_log_file(self) -> None:
        """Ensure log file exists with valid structure."""
        if not self._log_path.exists():
            self._write_log({"sessions": []})
            logger.info(f"Created request history log at {self._log_path}")

    def _read_log(self) -> dict:
        """Read log file contents.

        Returns:
            Log data as dictionary.
        """
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"sessions": []}

    def _write_log(self, data: dict) -> None:
        """Write log file contents.

        Args:
            data: Log data to write.
        """
        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _rotate_old_logs(self) -> None:
        """Remove log entries older than retention period."""
        with self._lock:
            log_data = self._read_log()
            cutoff = datetime.now() - timedelta(days=self._settings.log_retention_days)
            cutoff_str = cutoff.isoformat()

            original_count = len(log_data.get("sessions", []))
            log_data["sessions"] = [
                s for s in log_data.get("sessions", [])
                if s.get("timestamp", "") >= cutoff_str
            ]

            removed = original_count - len(log_data["sessions"])
            if removed > 0:
                self._write_log(log_data)
                logger.info(f"Rotated {removed} old log entries")

    def log_session(self, result: PipelineResult) -> None:
        """Log a complete pipeline session.

        Args:
            result: Pipeline result to log.
        """
        session_data = self._serialize_result(result)

        with self._lock:
            log_data = self._read_log()
            log_data["sessions"].append(session_data)
            self._write_log(log_data)

        logger.info(f"Logged session {result.session_id}")

    def _serialize_result(self, result: PipelineResult) -> dict:
        """Serialize pipeline result for JSON storage.

        Args:
            result: Pipeline result to serialize.

        Returns:
            JSON-serializable dictionary.
        """
        return {
            "id": result.session_id,
            "timestamp": datetime.now().isoformat(),
            "topic": result.topic,
            "success": result.success,
            "error": result.error,
            "total_duration_ms": result.total_duration_ms,
            "summary": {
                "total_sections": result.total_sections,
                "low_confidence_sections": result.low_confidence_sections,
            },
            "steps": [self._serialize_step(s) for s in result.steps],
            "deep_dives": [self._serialize_deep_dive(d) for d in result.deep_dive_results],
        }

    def _serialize_step(self, step: Any) -> dict:
        """Serialize a pipeline step.

        Args:
            step: PipelineStep to serialize.

        Returns:
            JSON-serializable dictionary.
        """
        return {
            "name": step.name,
            "started_at": step.started_at.isoformat() if step.started_at else None,
            "completed_at": step.completed_at.isoformat() if step.completed_at else None,
            "duration_ms": step.duration_ms,
            "success": step.success,
            "error": step.error,
            "metadata": step.metadata,
        }

    def _serialize_deep_dive(self, result: BarRaiserResult) -> dict:
        """Serialize a Bar Raiser result.

        Args:
            result: BarRaiserResult to serialize.

        Returns:
            JSON-serializable dictionary.
        """
        return {
            "topic": result.topic,
            "low_confidence": result.low_confidence,
            "total_duration_ms": result.total_duration_ms,
            "success": result.success,
            "error": result.error,
            "attempts": [self._serialize_attempt(a) for a in result.attempts],
            # Store truncated content for reference
            "final_content_preview": result.final_content[:500] if result.final_content else "",
        }

    def _serialize_attempt(self, attempt: AttemptRecord) -> dict:
        """Serialize a single attempt record.

        Args:
            attempt: AttemptRecord to serialize.

        Returns:
            JSON-serializable dictionary.
        """
        return {
            "attempt_number": attempt.attempt_number,
            "strictness": attempt.strictness.value,
            "critique_passed": attempt.critique_passed,
            "critique_feedback": attempt.critique_feedback,
            "draft_duration_ms": attempt.draft_duration_ms,
            "critique_duration_ms": attempt.critique_duration_ms,
            # Store truncated draft for traceability
            "draft_preview": attempt.draft[:500] if attempt.draft else "",
        }

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        """Get recent session summaries.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session summaries.
        """
        with self._lock:
            log_data = self._read_log()
            sessions = log_data.get("sessions", [])
            # Return most recent first
            return list(reversed(sessions[-limit:]))

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a specific session by ID.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            Session data or None if not found.
        """
        with self._lock:
            log_data = self._read_log()
            for session in log_data.get("sessions", []):
                if session.get("id") == session_id:
                    return session
        return None


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
