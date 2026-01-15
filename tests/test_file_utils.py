"""Tests for utils/file_utils.py."""

import os
from pathlib import Path

import pytest

from utils.file_utils import FileManager


class TestFileManager:
    """Tests for FileManager class."""

    def test_init_with_default_path(self):
        """Test initialization with default Cyrus path."""
        manager = FileManager()
        assert manager._cyrus_root is not None

    def test_init_with_custom_path(self, temp_dir):
        """Test initialization with custom path."""
        manager = FileManager(cyrus_root=str(temp_dir))
        assert manager._cyrus_root == temp_dir

    def test_is_cyrus_available_when_exists(self, mock_cyrus_root):
        """Test Cyrus availability check when directory exists."""
        manager = FileManager(cyrus_root=str(mock_cyrus_root))
        available, message = manager.is_cyrus_available()

        assert available is True
        assert "accessible" in message

    def test_is_cyrus_available_when_missing(self, temp_dir):
        """Test Cyrus availability check when directory is missing."""
        manager = FileManager(cyrus_root=str(temp_dir / "nonexistent"))
        available, message = manager.is_cyrus_available()

        assert available is False
        assert "not found" in message

    def test_ensure_responses_dir_creates_directory(self, mock_cyrus_root):
        """Test responses directory creation."""
        # Remove existing responses dir
        responses_dir = mock_cyrus_root / "gemini-responses"
        if responses_dir.exists():
            responses_dir.rmdir()

        manager = FileManager(cyrus_root=str(mock_cyrus_root))
        result = manager.ensure_responses_dir()

        assert result is True
        assert responses_dir.exists()

    def test_generate_filename_basic(self):
        """Test basic filename generation."""
        manager = FileManager()
        filename = manager.generate_filename("Test Title")

        assert filename.endswith(".md")
        assert "test-title" in filename

    def test_generate_filename_special_characters(self):
        """Test filename generation with special characters."""
        manager = FileManager()
        filename = manager.generate_filename("Test: A Complex Title! @#$%")

        assert ".md" in filename
        # Special characters should be removed
        assert ":" not in filename
        assert "@" not in filename

    def test_generate_filename_long_title(self):
        """Test filename generation with very long title."""
        manager = FileManager()
        long_title = "A" * 200
        filename = manager.generate_filename(long_title)

        # Filename (without extension and timestamp) should be reasonable length
        # Timestamp format is YYYYMMDD-HHMM (13 chars + dash = 14)
        base = filename.replace(".md", "")
        # Remove timestamp suffix (format: -YYYYMMDD-HHMM)
        base_without_timestamp = base.rsplit("-", 2)[0] if base.count("-") >= 2 else base
        assert len(base_without_timestamp) <= 80

    def test_extract_title_from_h1(self):
        """Test title extraction from H1 heading."""
        manager = FileManager()
        content = "# My Great Title\n\nSome content here."
        title = manager.extract_title(content)

        assert title == "My Great Title"

    def test_extract_title_fallback(self):
        """Test title extraction fallback to first line."""
        manager = FileManager()
        content = "Just some text without headers"
        title = manager.extract_title(content)

        assert title == "Just some text without headers"

    def test_extract_title_empty_content(self):
        """Test title extraction from empty content."""
        manager = FileManager()
        title = manager.extract_title("")

        assert title == "Untitled Guide"

    def test_save_guide_success(self, mock_cyrus_root):
        """Test successful guide saving."""
        manager = FileManager(cyrus_root=str(mock_cyrus_root))

        success, filepath, message = manager.save_guide(
            content="# Test Guide\n\nContent here.",
            low_confidence_count=0,
        )

        assert success is True
        assert filepath != ""
        assert Path(filepath).exists()

        # Check content
        with open(filepath, "r") as f:
            saved_content = f.read()
        assert "Test Guide" in saved_content
        assert "---" in saved_content  # YAML frontmatter

    def test_save_guide_with_low_confidence(self, mock_cyrus_root):
        """Test guide saving with low confidence sections."""
        manager = FileManager(cyrus_root=str(mock_cyrus_root))

        success, filepath, message = manager.save_guide(
            content="# Guide with Issues\n\nContent.",
            low_confidence_count=2,
        )

        assert success is True

        with open(filepath, "r") as f:
            saved_content = f.read()
        assert "low_confidence_sections: 2" in saved_content
        assert "review_recommended: true" in saved_content

    def test_save_guide_cyrus_not_available(self, temp_dir):
        """Test guide saving when Cyrus is not available."""
        manager = FileManager(cyrus_root=str(temp_dir / "nonexistent"))

        success, filepath, message = manager.save_guide(
            content="# Test\n\nContent",
        )

        assert success is False
        assert "not found" in message

    def test_list_saved_guides(self, mock_cyrus_root):
        """Test listing saved guides."""
        manager = FileManager(cyrus_root=str(mock_cyrus_root))

        # Save a couple of guides
        manager.save_guide("# Guide 1\n\nContent 1")
        manager.save_guide("# Guide 2\n\nContent 2")

        guides = manager.list_saved_guides()

        assert len(guides) >= 2
        assert all("filename" in g for g in guides)
        assert all("path" in g for g in guides)

    def test_list_saved_guides_empty(self, mock_cyrus_root):
        """Test listing guides when none exist."""
        # Clear the responses directory
        responses_dir = mock_cyrus_root / "gemini-responses"
        for f in responses_dir.glob("*.md"):
            f.unlink()

        manager = FileManager(cyrus_root=str(mock_cyrus_root))
        guides = manager.list_saved_guides()

        assert guides == []

    def test_read_guide(self, mock_cyrus_root):
        """Test reading a saved guide."""
        manager = FileManager(cyrus_root=str(mock_cyrus_root))

        # Save a guide first
        success, filepath, _ = manager.save_guide("# Test Read\n\nContent to read.")
        filename = Path(filepath).name

        content = manager.read_guide(filename)

        assert content is not None
        assert "Test Read" in content

    def test_read_guide_not_found(self, mock_cyrus_root):
        """Test reading non-existent guide."""
        manager = FileManager(cyrus_root=str(mock_cyrus_root))

        content = manager.read_guide("nonexistent.md")

        assert content is None

    def test_delete_guide(self, mock_cyrus_root):
        """Test deleting a saved guide."""
        manager = FileManager(cyrus_root=str(mock_cyrus_root))

        # Save then delete
        success, filepath, _ = manager.save_guide("# To Delete\n\nContent.")
        filename = Path(filepath).name

        result = manager.delete_guide(filename)

        assert result is True
        assert not Path(filepath).exists()

    def test_delete_guide_not_found(self, mock_cyrus_root):
        """Test deleting non-existent guide."""
        manager = FileManager(cyrus_root=str(mock_cyrus_root))

        result = manager.delete_guide("nonexistent.md")

        assert result is False


class TestFileManagerMetadata:
    """Tests for metadata generation."""

    def test_metadata_includes_required_fields(self, mock_cyrus_root):
        """Test that metadata includes all required fields."""
        manager = FileManager(cyrus_root=str(mock_cyrus_root))

        success, filepath, _ = manager.save_guide(
            content="# Test\n\nContent",
            low_confidence_count=1,
        )

        with open(filepath, "r") as f:
            content = f.read()

        assert "title:" in content
        assert "generated_at:" in content
        assert "source: Professor Gemini" in content
        assert "low_confidence_sections:" in content
