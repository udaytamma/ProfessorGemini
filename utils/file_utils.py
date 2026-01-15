"""File management utilities for Cyrus integration.

Handles saving Master Guides as Markdown files to the Cyrus
gemini-responses directory.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import get_settings


logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for Cyrus integration.

    Handles saving generated guides to the gemini-responses directory
    within the Cyrus project.
    """

    def __init__(self, cyrus_root: Optional[str] = None) -> None:
        """Initialize file manager.

        Args:
            cyrus_root: Optional custom Cyrus root path.
        """
        self._settings = get_settings()
        self._cyrus_root = Path(cyrus_root or self._settings.cyrus_root_path)
        self._responses_dir = self._cyrus_root / "gemini-responses"

    def is_cyrus_available(self) -> tuple[bool, str]:
        """Check if Cyrus project is accessible.

        Returns:
            Tuple of (available, message).
        """
        if not self._cyrus_root.exists():
            return False, f"Cyrus root not found: {self._cyrus_root}"

        if not self._cyrus_root.is_dir():
            return False, f"Cyrus root is not a directory: {self._cyrus_root}"

        return True, "Cyrus project accessible"

    def ensure_responses_dir(self) -> bool:
        """Ensure gemini-responses directory exists.

        Returns:
            True if directory exists or was created.
        """
        try:
            self._responses_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            logger.error(f"Failed to create responses directory: {e}")
            return False

    def generate_filename(self, title: str) -> str:
        """Generate a safe filename from title.

        Args:
            title: Title to convert to filename.

        Returns:
            Safe filename with .md extension.
        """
        # Clean title for filename
        # Remove special characters, keep alphanumeric and spaces
        clean = re.sub(r"[^\w\s-]", "", title)
        # Replace spaces with hyphens
        clean = re.sub(r"[\s_]+", "-", clean)
        # Convert to lowercase
        clean = clean.lower().strip("-")
        # Limit length
        clean = clean[:80]

        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")

        return f"{clean}-{timestamp}.md"

    def extract_title(self, content: str) -> str:
        """Extract title from markdown content.

        Looks for first H1 heading or first line.

        Args:
            content: Markdown content.

        Returns:
            Extracted title.
        """
        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                # H1 heading found
                title = line[2:].strip()
                # Remove common prefixes
                prefixes_to_remove = [
                    "The Principal TPM's Guide to ",
                    "A Principal TPM's Guide to ",
                    "Principal TPM's Guide to ",
                    "Guide to ",
                ]
                for prefix in prefixes_to_remove:
                    if title.startswith(prefix):
                        title = title[len(prefix):]
                        break
                return title

        # Fallback to first non-empty line
        for line in lines:
            line = line.strip()
            if line:
                return line[:100]

        return "Untitled Guide"

    def save_guide(
        self,
        content: str,
        title: Optional[str] = None,
        low_confidence_count: int = 0,
    ) -> tuple[bool, str, str]:
        """Save Master Guide to gemini-responses directory.

        Args:
            content: Markdown content to save.
            title: Optional title override.
            low_confidence_count: Number of low confidence sections.

        Returns:
            Tuple of (success, filepath, message).
        """
        # Check Cyrus availability
        available, message = self.is_cyrus_available()
        if not available:
            return False, "", message

        # Ensure directory exists
        if not self.ensure_responses_dir():
            return False, "", "Failed to create responses directory"

        # Extract or use provided title
        doc_title = title or self.extract_title(content)
        filename = self.generate_filename(doc_title)
        filepath = self._responses_dir / filename

        # Prepare content with metadata header
        metadata = self._generate_metadata(doc_title, low_confidence_count)
        full_content = f"{metadata}\n\n{content}"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_content)

            logger.info(f"Saved guide to {filepath}")
            return True, str(filepath), f"Saved to {filename}"

        except OSError as e:
            error_msg = f"Failed to save file: {e}"
            logger.error(error_msg)
            return False, "", error_msg

    def _generate_metadata(self, title: str, low_confidence_count: int) -> str:
        """Generate YAML frontmatter metadata.

        Args:
            title: Document title.
            low_confidence_count: Number of low confidence sections.

        Returns:
            YAML frontmatter string.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metadata = [
            "---",
            f"title: \"{title}\"",
            f"generated_at: \"{timestamp}\"",
            "source: Professor Gemini",
            f"low_confidence_sections: {low_confidence_count}",
        ]

        if low_confidence_count > 0:
            metadata.append("review_recommended: true")

        metadata.append("---")

        return "\n".join(metadata)

    def list_saved_guides(self) -> list[dict]:
        """List all saved guides in responses directory.

        Returns:
            List of guide metadata dictionaries.
        """
        if not self._responses_dir.exists():
            return []

        guides = []
        for filepath in sorted(self._responses_dir.glob("*.md"), reverse=True):
            stat = filepath.stat()
            guides.append({
                "filename": filepath.name,
                "path": str(filepath),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return guides

    def read_guide(self, filename: str) -> Optional[str]:
        """Read a saved guide by filename.

        Args:
            filename: Name of the file to read.

        Returns:
            File content or None if not found.
        """
        filepath = self._responses_dir / filename

        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.error(f"Failed to read guide {filename}: {e}")
            return None

    def delete_guide(self, filename: str) -> bool:
        """Delete a saved guide by filename.

        Args:
            filename: Name of the file to delete.

        Returns:
            True if deleted successfully.
        """
        filepath = self._responses_dir / filename

        if not filepath.exists():
            return False

        try:
            filepath.unlink()
            logger.info(f"Deleted guide: {filename}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete guide {filename}: {e}")
            return False
