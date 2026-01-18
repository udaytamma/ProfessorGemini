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

# Interview Questions section pattern
INTERVIEW_QUESTIONS_PATTERN = re.compile(
    r"^##\s*Interview\s+Questions?\s*$",
    re.MULTILINE | re.IGNORECASE
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


def extract_interview_questions(content: str) -> tuple[str, str]:
    """Extract interview questions section from content.

    Looks for "## Interview Questions" header and extracts everything after it.
    Returns the content without interview questions and the interview questions separately.

    Args:
        content: Section content that may contain interview questions.

    Returns:
        Tuple of (content_without_questions, interview_questions_text).
        If no interview questions found, returns (original_content, "").
    """
    match = INTERVIEW_QUESTIONS_PATTERN.search(content)
    if not match:
        return content, ""

    # Split at the interview questions header
    main_content = content[:match.start()].strip()
    questions_text = content[match.end():].strip()

    return main_content, questions_text


def remove_duplicate_headers(content: str, section_topic: str) -> str:
    """Remove duplicate section headers and fix subsection header levels.

    Professor Gemini deep dives may generate content with:
    1. A leading header matching the section topic (should be removed - added by synthesis)
    2. Duplicate Roman numeral headers (## I., ## II., etc.) from its own structure
    3. H1 headers with Roman numerals (orphaned from deep dive generation)
    4. Numbered H2 headers (## 1., ## 2.) that should be H3 subsections
    5. A *** separator followed by duplicate content

    This function cleans up these issues to produce properly structured content.

    Args:
        content: Section content that may have duplicate headers.
        section_topic: The section topic (e.g., "I. Architectural Fundamentals").

    Returns:
        Content with duplicate headers removed and subsections fixed.
    """
    processed = content.strip()

    # 1. Remove ANY leading header (H1 or H2) at the very start of content
    # This handles cases where Gemini ignores the "no header" instruction
    processed = re.sub(r"^#{1,2}\s+[^\n]+\n+", "", processed)

    # 2. Remove H1/H2 headers that match section_topic anywhere in content
    # Escape special regex chars in section_topic
    escaped_topic = re.escape(section_topic)
    section_header_pattern = re.compile(
        r"\n#{1,2}\s+" + escaped_topic + r"\s*\n",
        re.IGNORECASE
    )
    processed = section_header_pattern.sub("\n", processed)

    # 3. Remove ALL H1 headers with Roman numerals (these are orphaned duplicates)
    # Deep dives sometimes create their own # I., # II. structure
    processed = re.sub(
        r"\n#\s+[IVXLCDM]+\.\s*[^\n]+\n",
        "\n",
        processed
    )

    # 4. Remove *** separators and any content structure markers
    # These are artifacts from deep dive generation
    processed = re.sub(r"\n\*{3}\n*", "\n\n", processed)

    # 5. Convert numbered H2 headers (## 1., ## 2., etc.) to H3
    # Deep dives create their own subsection structure that conflicts with Roman numerals
    processed = re.sub(
        r"^(##)\s+(\d+\.)",
        r"###\2",
        processed,
        flags=re.MULTILINE
    )

    # 6. Remove any remaining duplicate headers at the SAME level
    # Only compare H2 vs H2, H3 vs H3 - don't strip H2 if there's a matching H3
    # This preserves main sections (## I.) even if Interview Questions has (### I.)
    lines = processed.split("\n")
    seen_headings: set[str] = set()  # "level:heading_text" format
    result_lines = []

    for line in lines:
        heading_match = re.match(r"^(#{1,3})\s+([IVXLCDM]+\.\s*.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip().lower()
            # Use level + text as key to only detect same-level duplicates
            key = f"{level}:{heading_text}"
            if key in seen_headings:
                # Skip duplicate heading at same level
                continue
            seen_headings.add(key)
        result_lines.append(line)

    processed = "\n".join(result_lines)

    # 7. Clean up excessive blank lines
    processed = re.sub(r"\n{4,}", "\n\n\n", processed)

    return processed.strip()


def synthesize_locally(
    sections: list[dict],
    topic: str = "",
) -> LocalSynthesisResult:
    """Synthesize sections into a Master Guide without calling Gemini.

    Concatenates sections in order with proper formatting.
    Interview questions are extracted from each section and consolidated
    into a single section at the end of the guide.

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
    all_interview_questions = []

    # Title
    if topic:
        parts.append(f"# {topic}\n")

    # Executive summary
    section_titles = [s.get("topic", "Section") for s in sections]
    parts.append(f"This guide covers {len(sections)} key areas: {', '.join(section_titles)}.\n")

    # Add each section (extracting interview questions and removing duplicate headers)
    for section in sections:
        section_topic = section.get("topic", "Section")
        section_content = section.get("content", "")
        low_confidence = section.get("low_confidence", False)

        # Remove duplicate headers from section content
        cleaned_content = remove_duplicate_headers(section_content, section_topic)

        # Extract interview questions from cleaned content
        main_content, questions = extract_interview_questions(cleaned_content)

        # Collect interview questions with section context
        if questions:
            all_interview_questions.append({
                "topic": section_topic,
                "questions": questions,
            })

        # Add section header
        if low_confidence:
            parts.append(f"\n## {section_topic} ⚠️\n")
            parts.append("*Note: This section may need additional review.*\n")
        else:
            parts.append(f"\n## {section_topic}\n")

        # Add section content (without interview questions and duplicate headers)
        parts.append(main_content)

    # Add consolidated Interview Questions section at the end
    if all_interview_questions:
        parts.append("\n---\n")
        parts.append("\n## Interview Questions\n")
        for iq in all_interview_questions:
            parts.append(f"\n### {iq['topic']}\n")
            parts.append(iq["questions"])

    # Conclusion
    parts.append("\n---\n")
    parts.append("\n## Key Takeaways\n")
    parts.append("- Review each section for actionable insights applicable to your organization\n")
    parts.append("- Consider the trade-offs discussed when making architectural decisions\n")
    parts.append("- Use the operational considerations as a checklist for production readiness\n")

    content = "\n".join(parts)

    logger.info(
        f"Local synthesis completed: {len(sections)} sections, "
        f"{len(all_interview_questions)} interview question sets, {len(content)} chars"
    )

    return LocalSynthesisResult(
        content=content,
        success=True,
    )
