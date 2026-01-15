"""Local processing utilities to reduce Gemini API calls.

Provides local alternatives for topic splitting and synthesis operations
that would otherwise require Gemini API calls.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


# Roman numeral pattern: matches "## I. ", "## II. ", etc.
ROMAN_SECTION_PATTERN = re.compile(
    r"^##\s+(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+(.+)$",
    re.MULTILINE
)


@dataclass
class LocalSplitResult:
    """Result from local topic splitting.

    Attributes:
        topics: List of extracted topic titles.
        sections: Dict mapping topic title to section content.
        success: Whether splitting succeeded.
        error: Error message if failed.
    """

    topics: list[str]
    sections: dict[str, str]
    success: bool = True
    error: Optional[str] = None


def split_by_roman_numerals(content: str) -> LocalSplitResult:
    """Split content into sections based on Roman numeral headers.

    Parses content looking for "## I. Title", "## II. Title" patterns
    and extracts each section's content.

    Args:
        content: The base knowledge content with Roman numeral sections.

    Returns:
        LocalSplitResult with topics and section content.
    """
    # Find all Roman numeral section headers
    matches = list(ROMAN_SECTION_PATTERN.finditer(content))

    if not matches:
        logger.warning("No Roman numeral sections found in content")
        return LocalSplitResult(
            topics=[],
            sections={},
            success=False,
            error="No Roman numeral sections (## I., ## II., etc.) found in content",
        )

    topics = []
    sections = {}

    for i, match in enumerate(matches):
        numeral = match.group(1)
        title = match.group(2).strip()
        full_title = f"{numeral}. {title}"

        # Get content between this header and the next (or end of content)
        start_pos = match.end()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)

        section_content = content[start_pos:end_pos].strip()

        topics.append(full_title)
        sections[full_title] = section_content

    logger.info(f"Local split extracted {len(topics)} sections: {topics}")

    return LocalSplitResult(
        topics=topics,
        sections=sections,
        success=True,
    )


@dataclass
class LocalSynthesisResult:
    """Result from local synthesis.

    Attributes:
        content: The synthesized Master Guide content.
        success: Whether synthesis succeeded.
        error: Error message if failed.
    """

    content: str
    success: bool = True
    error: Optional[str] = None


def synthesize_locally(
    sections: list[dict],
    topic: str = "",
) -> LocalSynthesisResult:
    """Synthesize sections into a Master Guide without calling Gemini.

    Simply concatenates sections in order with proper formatting.
    Sections are expected to already be in the correct order from
    the Roman numeral split.

    Args:
        sections: List of dicts with 'topic', 'content', and optional 'low_confidence' keys.
        topic: The main topic for the guide title.

    Returns:
        LocalSynthesisResult with the synthesized content.
    """
    if not sections:
        return LocalSynthesisResult(
            content="",
            success=False,
            error="No sections to synthesize",
        )

    # Build the Master Guide
    parts = []

    # Title
    if topic:
        parts.append(f"# {topic}\n")

    # Executive summary
    section_titles = [s.get("topic", "Section") for s in sections]
    parts.append(f"This guide covers {len(sections)} key areas: {', '.join(section_titles)}.\n")

    # Add each section
    for section in sections:
        section_topic = section.get("topic", "Section")
        section_content = section.get("content", "")
        low_confidence = section.get("low_confidence", False)

        # Add section header
        if low_confidence:
            parts.append(f"\n## {section_topic} ⚠️\n")
            parts.append("*Note: This section may need additional review.*\n")
        else:
            parts.append(f"\n## {section_topic}\n")

        # Add section content
        parts.append(section_content)

    # Conclusion
    parts.append("\n---\n")
    parts.append("\n## Key Takeaways\n")
    parts.append("- Review each section for actionable insights applicable to your organization\n")
    parts.append("- Consider the trade-offs discussed when making architectural decisions\n")
    parts.append("- Use the operational considerations as a checklist for production readiness\n")

    content = "\n".join(parts)

    logger.info(f"Local synthesis completed: {len(sections)} sections, {len(content)} chars")

    return LocalSynthesisResult(
        content=content,
        success=True,
    )
