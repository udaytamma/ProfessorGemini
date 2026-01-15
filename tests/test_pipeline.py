"""Tests for core/pipeline.py."""

import os
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest

from core.pipeline import Pipeline, PipelineResult, PipelineStep
from core.gemini_client import GeminiResponse
from core.bar_raiser import BarRaiserResult


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_create_step(self):
        """Test creating a pipeline step."""
        step = PipelineStep(
            name="test_step",
            started_at=datetime.now(),
            duration_ms=1000,
            success=True,
        )

        assert step.name == "test_step"
        assert step.success is True
        assert step.error is None

    def test_step_with_error(self):
        """Test creating a failed step."""
        step = PipelineStep(
            name="failed_step",
            success=False,
            error="Something went wrong",
        )

        assert step.success is False
        assert step.error == "Something went wrong"


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful pipeline result."""
        result = PipelineResult(
            session_id="test-123",
            topic="Test Topic",
            master_guide="# Guide\n\nContent",
            low_confidence_sections=0,
            total_sections=3,
            success=True,
        )

        assert result.success is True
        assert result.total_sections == 3

    def test_failed_result(self):
        """Test creating a failed pipeline result."""
        result = PipelineResult(
            session_id="test-456",
            topic="Failed Topic",
            master_guide="",
            low_confidence_sections=0,
            total_sections=0,
            success=False,
            error="API key missing",
        )

        assert result.success is False
        assert result.error == "API key missing"


class TestPipeline:
    """Tests for Pipeline class."""

    @pytest.fixture
    def mock_pipeline(self, mock_env_vars):
        """Create a pipeline with mocked clients."""
        # Set USE_CLAUDE to true for hybrid mode tests
        with patch.dict(os.environ, {"USE_CLAUDE": "true"}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            with patch("core.pipeline.GeminiClient") as mock_gemini:
                with patch("anthropic.Anthropic"):  # Mock Anthropic client
                    mock_gemini_instance = MagicMock()
                    mock_gemini.return_value = mock_gemini_instance

                    # Import ClaudeClient after mocking
                    from core.claude_client import ClaudeClient
                    with patch.object(ClaudeClient, '__init__', lambda self: None):
                        with patch.object(ClaudeClient, 'is_available', return_value=True):
                            pipeline = Pipeline()

                            # Store mocks for test access
                            pipeline._gemini = mock_gemini_instance
                            pipeline._claude = MagicMock()

                            yield pipeline

    @pytest.fixture
    def mock_pipeline_gemini_only(self, mock_env_vars):
        """Create a pipeline in Gemini-only mode."""
        with patch.dict(os.environ, {"USE_CLAUDE": "false"}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            with patch("core.pipeline.GeminiClient") as mock_gemini:
                mock_gemini_instance = MagicMock()
                mock_gemini.return_value = mock_gemini_instance

                pipeline = Pipeline()
                pipeline._gemini = mock_gemini_instance

                yield pipeline

    def test_is_ready_both_configured(self, mock_pipeline):
        """Test is_ready when both clients available."""
        mock_pipeline._gemini.is_available.return_value = True
        mock_pipeline._claude.is_available.return_value = True

        ready, message = mock_pipeline.is_ready()

        assert ready is True
        assert "ready" in message.lower()

    def test_is_ready_gemini_missing(self, mock_pipeline):
        """Test is_ready when Gemini not configured."""
        mock_pipeline._gemini.is_available.return_value = False
        mock_pipeline._claude.is_available.return_value = True

        ready, message = mock_pipeline.is_ready()

        assert ready is False
        assert "Gemini" in message

    def test_is_ready_claude_missing(self, mock_pipeline):
        """Test is_ready when Claude not configured."""
        mock_pipeline._gemini.is_available.return_value = True
        mock_pipeline._claude.is_available.return_value = False

        ready, message = mock_pipeline.is_ready()

        assert ready is False
        assert "Claude" in message

    def test_is_ready_gemini_only_mode(self, mock_pipeline_gemini_only):
        """Test is_ready in Gemini-only mode."""
        mock_pipeline_gemini_only._gemini.is_available.return_value = True

        ready, message = mock_pipeline_gemini_only.is_ready()

        assert ready is True
        assert "Gemini-only" in message

    def test_execute_not_ready(self, mock_pipeline):
        """Test execute fails when not ready."""
        mock_pipeline._gemini.is_available.return_value = False
        mock_pipeline._claude.is_available.return_value = False

        result = mock_pipeline.execute("Test topic")

        assert result.success is False
        assert "not configured" in result.error

    def test_execute_step1_failure(self, mock_pipeline):
        """Test execute handles Step 1 failure."""
        mock_pipeline._gemini.is_available.return_value = True
        mock_pipeline._claude.is_available.return_value = True

        mock_pipeline._gemini.generate_base_knowledge.return_value = GeminiResponse(
            content="",
            model="gemini-3-pro-preview",
            duration_ms=100,
            success=False,
            error="API Error",
        )

        result = mock_pipeline.execute("Test topic")

        assert result.success is False
        assert "Step 1" in result.error

    def test_execute_step2_failure(self, mock_pipeline):
        """Test execute handles Step 2 failure."""
        from core.claude_client import ClaudeResponse

        mock_pipeline._gemini.is_available.return_value = True
        mock_pipeline._claude.is_available.return_value = True

        mock_pipeline._gemini.generate_base_knowledge.return_value = GeminiResponse(
            content="Base knowledge content",
            model="gemini-3-pro-preview",
            duration_ms=1000,
            success=True,
        )

        mock_pipeline._claude.split_into_topics.return_value = (
            [],
            ClaudeResponse(
                content="",
                model="claude-opus-4-5-20251101",
                duration_ms=500,
                success=False,
                error="Failed to split",
            ),
        )

        result = mock_pipeline.execute("Test topic")

        assert result.success is False
        assert "Step 2" in result.error

    @patch("core.pipeline.BarRaiser")
    @patch("core.pipeline.ThreadPoolExecutor")
    def test_execute_full_success(self, mock_executor, mock_bar_raiser, mock_pipeline):
        """Test successful full pipeline execution."""
        from core.claude_client import ClaudeResponse

        mock_pipeline._gemini.is_available.return_value = True
        mock_pipeline._claude.is_available.return_value = True

        # Step 1: Base knowledge
        mock_pipeline._gemini.generate_base_knowledge.return_value = GeminiResponse(
            content="Comprehensive base knowledge",
            model="gemini-3-pro-preview",
            duration_ms=2000,
            success=True,
        )

        # Step 2: Split topics
        mock_pipeline._claude.split_into_topics.return_value = (
            ["Topic 1", "Topic 2", "Topic 3"],
            ClaudeResponse(
                content='["Topic 1", "Topic 2", "Topic 3"]',
                model="claude-opus-4-5-20251101",
                duration_ms=500,
                success=True,
            ),
        )

        # Step 3: Bar Raiser results
        mock_bar_raiser_instance = MagicMock()
        mock_bar_raiser_instance.process_topics_parallel.return_value = [
            BarRaiserResult(
                topic="Topic 1",
                final_content="Content 1",
                low_confidence=False,
                total_duration_ms=1000,
                success=True,
            ),
            BarRaiserResult(
                topic="Topic 2",
                final_content="Content 2",
                low_confidence=False,
                total_duration_ms=1000,
                success=True,
            ),
            BarRaiserResult(
                topic="Topic 3",
                final_content="Content 3",
                low_confidence=True,  # One low confidence
                total_duration_ms=2000,
                success=True,
            ),
        ]
        mock_bar_raiser.return_value = mock_bar_raiser_instance

        # Step 4: Synthesis
        mock_pipeline._claude.synthesize_guide.return_value = ClaudeResponse(
            content="# Master Guide\n\n## Topic 1\n\nContent...",
            model="claude-opus-4-5-20251101",
            duration_ms=3000,
            success=True,
        )

        result = mock_pipeline.execute("Test topic")

        assert result.success is True
        assert result.total_sections == 3
        assert result.low_confidence_sections == 1
        assert len(result.master_guide) > 0  # Local synthesis produces content

    @patch("core.pipeline.BarRaiser")
    @patch("core.pipeline.ThreadPoolExecutor")
    def test_execute_no_sections_to_synthesize(self, mock_executor, mock_bar_raiser, mock_pipeline):
        """Test execute when all deep dives fail."""
        from core.claude_client import ClaudeResponse

        mock_pipeline._gemini.is_available.return_value = True
        mock_pipeline._claude.is_available.return_value = True

        mock_pipeline._gemini.generate_base_knowledge.return_value = GeminiResponse(
            content="Base content",
            model="gemini-3-pro-preview",
            duration_ms=1000,
            success=True,
        )

        mock_pipeline._claude.split_into_topics.return_value = (
            ["Topic 1"],
            ClaudeResponse(content='["Topic 1"]', model="claude-opus-4-5-20251101", duration_ms=500, success=True),
        )

        mock_bar_raiser_instance = MagicMock()
        mock_bar_raiser_instance.process_topics_parallel.return_value = [
            BarRaiserResult(
                topic="Topic 1",
                final_content="",  # No content
                low_confidence=True,
                success=False,
                error="Failed",
            ),
        ]
        mock_bar_raiser.return_value = mock_bar_raiser_instance

        result = mock_pipeline.execute("Test topic")

        assert result.success is False
        assert "No sections to synthesize" in result.error

    def test_status_callback_called(self, mock_env_vars):
        """Test that status callback is called during execution."""
        with patch.dict(os.environ, {"USE_CLAUDE": "false"}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            status_messages = []

            def capture_status(msg):
                status_messages.append(msg)

            with patch("core.pipeline.GeminiClient") as mock_gemini:
                mock_gemini_instance = MagicMock()
                mock_gemini.return_value = mock_gemini_instance
                mock_gemini_instance.is_available.return_value = True

                # Set up minimal mocks so pipeline can execute
                mock_gemini_instance.generate_base_knowledge.return_value = GeminiResponse(
                    content="Base",
                    model="gemini-3-pro-preview",
                    duration_ms=100,
                    success=True,
                )
                mock_gemini_instance.split_into_topics.return_value = (
                    [],  # Empty topics will cause early exit but status callback should still be called
                    GeminiResponse(content="[]", model="gemini-3-pro-preview", duration_ms=100, success=True),
                )

                # Pass status_callback during construction
                pipeline = Pipeline(status_callback=capture_status)
                pipeline.execute("Test topic")

                # Status callback should be called at least for steps 1 and 2
                assert len(status_messages) > 0


class TestPipelineGeminiOnly:
    """Tests for Pipeline in Gemini-only mode."""

    @pytest.fixture
    def mock_pipeline(self, mock_env_vars):
        """Create a pipeline in Gemini-only mode."""
        with patch.dict(os.environ, {"USE_CLAUDE": "false"}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            with patch("core.pipeline.GeminiClient") as mock_gemini:
                mock_gemini_instance = MagicMock()
                mock_gemini.return_value = mock_gemini_instance

                pipeline = Pipeline()
                pipeline._gemini = mock_gemini_instance

                yield pipeline

    def test_is_ready_gemini_only(self, mock_pipeline):
        """Test is_ready in Gemini-only mode."""
        mock_pipeline._gemini.is_available.return_value = True

        ready, message = mock_pipeline.is_ready()

        assert ready is True
        assert "Gemini-only" in message

    @patch("core.pipeline.BarRaiser")
    def test_execute_gemini_only_success(self, mock_bar_raiser, mock_pipeline):
        """Test successful execution in Gemini-only mode."""
        mock_pipeline._gemini.is_available.return_value = True

        # Step 1: Base knowledge
        mock_pipeline._gemini.generate_base_knowledge.return_value = GeminiResponse(
            content="Base knowledge",
            model="gemini-3-pro-preview",
            duration_ms=1000,
            success=True,
        )

        # Step 2: Split topics (Gemini)
        mock_pipeline._gemini.split_into_topics.return_value = (
            ["Topic 1", "Topic 2"],
            GeminiResponse(
                content='["Topic 1", "Topic 2"]',
                model="gemini-3-pro-preview",
                duration_ms=500,
                success=True,
            ),
        )

        # Step 3: Bar Raiser - mock the async method for Gemini-only mode
        mock_bar_raiser_instance = MagicMock()

        async def mock_process_topics_async(*args, **kwargs):
            return [
                BarRaiserResult(
                    topic="Topic 1",
                    final_content="Content 1",
                    low_confidence=False,
                    total_duration_ms=1000,
                    success=True,
                ),
                BarRaiserResult(
                    topic="Topic 2",
                    final_content="Content 2",
                    low_confidence=False,
                    total_duration_ms=1000,
                    success=True,
                ),
            ]

        mock_bar_raiser_instance.process_topics_async = mock_process_topics_async
        mock_bar_raiser.return_value = mock_bar_raiser_instance

        # Step 4: Synthesis (Gemini)
        mock_pipeline._gemini.synthesize_guide.return_value = GeminiResponse(
            content="# Master Guide\n\nSynthesized content",
            model="gemini-3-pro-preview",
            duration_ms=2000,
            success=True,
        )

        result = mock_pipeline.execute("Test topic")

        assert result.success is True
        assert result.total_sections == 2
        assert len(result.master_guide) > 0  # Local synthesis produces content


class TestPipelineIntegration:
    """Integration-style tests for pipeline."""

    def test_session_id_is_unique(self, mock_env_vars):
        """Test that each execution gets a unique session ID."""
        with patch.dict(os.environ, {"USE_CLAUDE": "false"}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            with patch("core.pipeline.GeminiClient") as mock_gemini:
                mock_gemini.return_value.is_available.return_value = False

                pipeline = Pipeline()

                result1 = pipeline.execute("Topic 1")
                result2 = pipeline.execute("Topic 2")

                assert result1.session_id != result2.session_id

    def test_steps_are_recorded(self, mock_env_vars):
        """Test that pipeline steps are recorded in result."""
        with patch.dict(os.environ, {"USE_CLAUDE": "false"}, clear=False):
            from config.settings import refresh_settings
            refresh_settings()

            with patch("core.pipeline.GeminiClient") as mock_gemini:
                mock_gemini_instance = MagicMock()
                mock_gemini.return_value = mock_gemini_instance

                mock_gemini_instance.is_available.return_value = True
                mock_gemini_instance.generate_base_knowledge.return_value = GeminiResponse(
                    content="",
                    model="gemini-3-pro-preview",
                    duration_ms=100,
                    success=False,
                    error="Failed",
                )

                pipeline = Pipeline()
                result = pipeline.execute("Test")

                assert len(result.steps) == 1
                assert result.steps[0].name == "base_knowledge"
                assert result.steps[0].success is False
