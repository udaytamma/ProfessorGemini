"""Tests for core/local_processing.py."""

from core.local_processing import (
    split_by_roman_numerals,
    extract_interview_questions,
    synthesize_locally,
)


class TestSplitByRomanNumerals:
    """Tests for local Roman numeral splitting."""

    def test_split_success(self):
        content = (
            "## I. First Section\n"
            "Content A\n\n"
            "## II. Second Section\n"
            "Content B\n"
        )

        result = split_by_roman_numerals(content)

        assert result.success is True
        assert result.topics == ["I. First Section", "II. Second Section"]
        assert "Content A" in result.sections["I. First Section"]

    def test_split_failure_when_missing_headers(self):
        content = "No roman numeral headers here."

        result = split_by_roman_numerals(content)

        assert result.success is False
        assert result.topics == []


class TestInterviewQuestionsExtraction:
    """Tests for interview questions extraction."""

    def test_extract_interview_questions(self):
        content = (
            "Main content\n\n"
            "## Interview Questions\n"
            "- Q1?\n- Q2?\n"
        )

        main, questions = extract_interview_questions(content)

        assert "Main content" in main
        assert "Q1" in questions


class TestLocalSynthesis:
    """Tests for local synthesis."""

    def test_synthesize_includes_questions(self):
        sections = [
            {
                "topic": "I. Topic One",
                "content": "Content one\n\n## Interview Questions\n- Q1?",
                "low_confidence": False,
            },
            {
                "topic": "II. Topic Two",
                "content": "Content two",
                "low_confidence": True,
            },
        ]

        result = synthesize_locally(sections, topic="Test Topic")

        assert result.success is True
        assert "# Test Topic" in result.content
        assert "## Interview Questions" in result.content
        assert "Q1" in result.content
