"""Knowledge Base context loader for Single Prompt mode.

Loads all markdown files from Cyrus gemini-responses directory
and formats them as context for Gemini API calls.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LoadedContext:
    """Result from loading Knowledge Base context."""

    content: str
    file_count: int
    total_chars: int
    success: bool
    error: Optional[str] = None
    rag_duration_ms: Optional[int] = None  # RAG retrieval timing (if used)


class ContextLoader:
    """Loads Knowledge Base documents as context for Gemini."""

    # Pattern to match YAML frontmatter: ---\n...\n---
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

    def __init__(self, gemini_responses_path: str) -> None:
        """Initialize with path to gemini-responses directory.

        Args:
            gemini_responses_path: Path to the gemini-responses directory.
        """
        self._responses_dir = Path(gemini_responses_path)

    def load_all_documents(self) -> LoadedContext:
        """Load all .md files and concatenate as context.

        Strips YAML frontmatter and concatenates with document separators.

        Returns:
            LoadedContext with concatenated content and metadata.
        """
        if not self._responses_dir.exists():
            return LoadedContext(
                content="",
                file_count=0,
                total_chars=0,
                success=False,
                error=f"Directory not found: {self._responses_dir}",
            )

        if not self._responses_dir.is_dir():
            return LoadedContext(
                content="",
                file_count=0,
                total_chars=0,
                success=False,
                error=f"Path is not a directory: {self._responses_dir}",
            )

        md_files = sorted(self._responses_dir.glob("*.md"))

        if not md_files:
            return LoadedContext(
                content="",
                file_count=0,
                total_chars=0,
                success=False,
                error=f"No markdown files found in: {self._responses_dir}",
            )

        documents = []
        for md_file in md_files:
            try:
                raw_content = md_file.read_text(encoding="utf-8")
                clean_content = self._strip_frontmatter(raw_content)

                # Add document separator with filename
                doc_section = f"--- Document: {md_file.name} ---\n{clean_content}"
                documents.append(doc_section)

            except Exception as e:
                logger.warning(f"Failed to read {md_file.name}: {e}")
                continue

        if not documents:
            return LoadedContext(
                content="",
                file_count=0,
                total_chars=0,
                success=False,
                error="All files failed to load",
            )

        combined_content = "\n\n".join(documents)

        logger.info(
            f"Loaded {len(documents)} documents, {len(combined_content):,} chars"
        )

        return LoadedContext(
            content=combined_content,
            file_count=len(documents),
            total_chars=len(combined_content),
            success=True,
        )

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content.

        Args:
            content: Raw markdown content with potential frontmatter.

        Returns:
            Content with frontmatter removed.
        """
        return self.FRONTMATTER_PATTERN.sub("", content).strip()
