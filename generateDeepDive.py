#!/usr/bin/env python3
"""generateDeepDive - Batch Knowledge Base article generator.

Extracts subsection topics from the Cyrus System Design guide and generates
comprehensive articles using Professor Gemini's Pipeline. Automatically updates
the Knowledge Base sidebar with a new collapsible section.

Usage:
    python generateDeepDive.py 2.6           # Generate Section 2.6 (Communication Patterns)
    python generateDeepDive.py 2.5           # Generate Section 2.5 (Migration Patterns)
    python generateDeepDive.py 3.1           # Generate Section 3.1 (Distributed Consensus)
    python generateDeepDive.py 2.6 --dry-run # Preview topics without generating

The script:
1. Parses the guide TSX file to extract <Subsection title="..."> tags for the section
2. Gets the section name (e.g., "Communication Patterns") for sidebar categorization
3. Runs Pipeline.execute() in parallel using ThreadPoolExecutor (max 5 workers)
4. Saves each result to gemini-responses/ via FileManager.save_guide()
5. Updates Knowledge Base page.tsx with new sidebar section (idempotent)

See SCRIPTS.md for full documentation.
"""

import argparse
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from config.settings import get_settings
from core.pipeline import Pipeline
from utils.file_utils import FileManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


GUIDE_PATH = Path("/Users/omega/Projects/Cyrus/src/app/nebula/system-design/guide/page.tsx")
KNOWLEDGE_BASE_PATH = Path("/Users/omega/Projects/Cyrus/src/app/nebula/knowledge-base/page.tsx")

MAX_WORKERS = 5


class TopicResult(NamedTuple):
    """Result from processing a single topic."""

    topic: str
    success: bool
    filepath: str
    slug: str
    error: str


def title_to_slug(title: str) -> str:
    """Convert a title to a URL-friendly slug.

    Matches the logic in sync-nebula-docs.js for consistency.
    """
    slug = title.lower()
    # Replace special characters and spaces with hyphens
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug


def get_section_name(section_number: str) -> str:
    """Extract the section name/title from the guide TSX.

    Args:
        section_number: Section number like "2.6" or "3.1"

    Returns:
        The section title (e.g., "Communication Patterns")
    """
    if not GUIDE_PATH.exists():
        raise FileNotFoundError(f"Guide file not found: {GUIDE_PATH}")

    content = GUIDE_PATH.read_text()

    # Convert "2.6" to "section-2-6" for ID matching
    parts = section_number.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid section number format: {section_number}")

    section_id = f"section-{parts[0]}-{parts[1]}"

    # Look for the Section component with this ID
    # Pattern: <Section id="section-2-6" title="Communication Patterns"
    section_pattern = rf'<Section[^>]*id="{section_id}"[^>]*title="([^"]+)"'
    match = re.search(section_pattern, content)

    if match:
        return match.group(1)

    # Alternative: look for the div with ID and find the nearest SectionTitle
    # Pattern: <div id="section-2-6"...><SectionTitle>2.6 Communication Patterns</SectionTitle>
    div_pattern = rf'<div id="{section_id}"[^>]*>.*?<SectionTitle[^>]*>[\d.]+\s*([^<]+)</SectionTitle>'
    match = re.search(div_pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()

    # Fallback: try to find any heading after the section div
    fallback_pattern = rf'<div id="{section_id}"[^>]*>.*?<h\d[^>]*>[\d.]*\s*([^<]+)</h\d>'
    match = re.search(fallback_pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()

    raise ValueError(f"Could not find section name for {section_number}")


def update_knowledge_base_sidebar(section_name: str, slugs: list[str]) -> bool:
    """Update the Knowledge Base page.tsx to add a new sidebar section.

    Args:
        section_name: The section name (e.g., "Communication Patterns")
        slugs: List of document slugs to include in this section

    Returns:
        True if update was successful, False otherwise
    """
    if not KNOWLEDGE_BASE_PATH.exists():
        logger.error(f"Knowledge Base page not found: {KNOWLEDGE_BASE_PATH}")
        return False

    content = KNOWLEDGE_BASE_PATH.read_text()

    # Generate variable names from section name
    # "Communication Patterns" -> "COMMUNICATION_PATTERNS_SLUGS", "communication"
    const_name = section_name.upper().replace(" ", "_").replace("-", "_") + "_SLUGS"
    state_key = section_name.lower().replace(" ", "").replace("-", "")

    # Check if this section already exists
    if const_name in content:
        logger.info(f"Section '{section_name}' already exists in sidebar, skipping update")
        return True

    # 1. Add the new SLUGS Set after the last existing one
    # Find the last SLUGS Set declaration
    slugs_pattern = r'(const \w+_SLUGS = new Set\(\[[^\]]+\]\);)'
    slugs_matches = list(re.finditer(slugs_pattern, content))

    if not slugs_matches:
        logger.error("Could not find existing SLUGS declarations in page.tsx")
        return False

    last_slugs_match = slugs_matches[-1]
    slugs_str = ",\n  ".join(f'"{slug}"' for slug in slugs)
    new_const = f'\n\nconst {const_name} = new Set([\n  {slugs_str},\n]);'

    insert_pos = last_slugs_match.end()
    content = content[:insert_pos] + new_const + content[insert_pos:]

    # 2. Update sectionsOpen state to include new section
    # Pattern: const [sectionsOpen, setSectionsOpen] = useState({ ... })
    state_pattern = r'(const \[sectionsOpen, setSectionsOpen\] = useState\(\{[^}]+)'
    state_match = re.search(state_pattern, content)

    if state_match:
        # Add new state key before the closing brace
        old_state = state_match.group(1)
        # Check if it ends with a comma
        if old_state.rstrip().endswith(","):
            new_state = old_state + f"\n    {state_key}: false,"
        else:
            new_state = old_state.rstrip() + f",\n    {state_key}: false,"
        content = content.replace(old_state, new_state)

    # 3. Add filter for new section docs and update otherDocs
    # Find the filter declarations
    filter_pattern = r'(const otherDocs = knowledgeBaseDocs\.filter\(\(doc\) => )([^;]+);'
    filter_match = re.search(filter_pattern, content)

    if filter_match:
        # Add new filter before otherDocs
        filter_prefix = filter_match.group(1)
        filter_conditions = filter_match.group(2)

        # Create variable name for the filtered docs
        docs_var_name = state_key + "Docs"

        # Add new filter line before otherDocs
        new_filter = f"const {docs_var_name} = knowledgeBaseDocs.filter((doc) => {const_name}.has(doc.slug));\n  "

        # Update otherDocs to exclude the new section
        new_conditions = filter_conditions.rstrip(")") + f" && !{const_name}.has(doc.slug))"

        # Find the line with otherDocs and insert new filter before it
        old_other_docs = filter_prefix + filter_conditions + ";"
        new_other_docs = new_filter + filter_prefix + new_conditions + ";"
        content = content.replace(old_other_docs, new_other_docs)

    # 4. Add the JSX for the new collapsible section
    # Find the pattern where we insert (before {otherDocs.length > 0 && (
    other_docs_jsx_pattern = r'(\{otherDocs\.length > 0 && \()'
    other_docs_match = re.search(other_docs_jsx_pattern, content)

    if other_docs_match:
        # Generate the JSX for the new section
        docs_var_name = state_key + "Docs"
        new_section_jsx = f'''{{{docs_var_name}.length > 0 && (
                <div>
                  <button
                    type="button"
                    onClick={{() =>
                      setSectionsOpen((prev) => ({{ ...prev, {state_key}: !prev.{state_key} }}))
                    }}
                    className="w-full px-2 py-1 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <span>{section_name}</span>
                    <span className="text-xs">{{sectionsOpen.{state_key} ? "▾" : "▸"}}</span>
                  </button>
                  {{sectionsOpen.{state_key} && (
                    <div className="space-y-1 mt-1">
                      {{{docs_var_name}.map((doc) => (
                        <button
                          key={{doc.slug}}
                          onClick={{() => handleDocSelect(doc.slug)}}
                          className={{`w-full text-left px-3 py-2.5 rounded-lg transition-colors ${{
                            selectedSlug === doc.slug
                              ? "bg-primary/10 text-primary border border-primary/30"
                              : "hover:bg-muted text-foreground"
                          }}`}}
                        >
                          <div className="font-medium text-sm truncate">{{doc.title}}</div>
                          <div className="text-xs text-muted-foreground mt-0.5">
                            {{formatDate(doc.date)}}
                          </div>
                        </button>
                      ))}}
                    </div>
                  )}}
                </div>
              )}}

              '''

        content = content.replace(
            other_docs_match.group(1),
            new_section_jsx + other_docs_match.group(1)
        )

    # Write the updated content
    KNOWLEDGE_BASE_PATH.write_text(content)
    logger.info(f"Updated Knowledge Base sidebar with '{section_name}' section")

    return True


def parse_section_topics(section_number: str) -> list[str]:
    """Extract subsection titles from the guide TSX for a given section.

    Args:
        section_number: Section number like "2.6" or "3.1"

    Returns:
        List of subsection titles found in that section.
    """
    if not GUIDE_PATH.exists():
        raise FileNotFoundError(f"Guide file not found: {GUIDE_PATH}")

    content = GUIDE_PATH.read_text()

    # Convert "2.6" to "section-2-6" for ID matching
    parts = section_number.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid section number format: {section_number} (expected X.Y)")

    section_id = f"section-{parts[0]}-{parts[1]}"

    # Find the section div start
    section_pattern = rf'<div id="{section_id}"[^>]*>'
    section_match = re.search(section_pattern, content)

    if not section_match:
        raise ValueError(f"Section {section_number} (id={section_id}) not found in guide")

    section_start = section_match.start()

    # Find the next section (next <div id="section-X-Y" or end of part)
    next_section_pattern = r'<div id="section-\d+-\d+"[^>]*>'
    next_match = re.search(next_section_pattern, content[section_match.end() :])

    if next_match:
        section_end = section_match.end() + next_match.start()
    else:
        section_end = len(content)

    section_content = content[section_start:section_end]

    # Extract all <Subsection title="..."> tags
    subsection_pattern = r'<Subsection title="([^"]+)"'
    subsection_titles = re.findall(subsection_pattern, section_content)

    return subsection_titles


def generate_topic(
    topic: str,
    file_manager: FileManager,
    worker_id: int,
) -> TopicResult:
    """Generate a guide for a single topic.

    Args:
        topic: The topic title to generate content for.
        file_manager: FileManager instance for saving.
        worker_id: Worker ID for logging.

    Returns:
        TopicResult with success status, filepath, and slug.
    """
    logger.info(f"[Worker {worker_id}] Starting: {topic}")

    # Generate slug for this topic
    slug = title_to_slug(topic)

    # Create pipeline with status callback for logging
    def status_callback(msg: str) -> None:
        logger.debug(f"[Worker {worker_id}] {msg}")

    pipeline = Pipeline(status_callback=status_callback)

    # Execute pipeline
    result = pipeline.execute(topic)

    if not result.success:
        logger.error(f"[Worker {worker_id}] Failed: {topic} - {result.error}")
        return TopicResult(
            topic=topic,
            success=False,
            filepath="",
            slug=slug,
            error=result.error or "Unknown error",
        )

    # Save the guide
    success, filepath, message = file_manager.save_guide(
        content=result.master_guide,
        title=topic,
        low_confidence_count=result.low_confidence_sections,
    )

    if success:
        logger.info(f"[Worker {worker_id}] Completed: {topic} -> {Path(filepath).name}")
        return TopicResult(
            topic=topic,
            success=True,
            filepath=filepath,
            slug=slug,
            error="",
        )
    else:
        logger.error(f"[Worker {worker_id}] Save failed: {topic} - {message}")
        return TopicResult(
            topic=topic,
            success=False,
            filepath="",
            slug=slug,
            error=message,
        )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Knowledge Base articles for System Design guide sections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generateDeepDive.py 2.6           # Communication Patterns
    python generateDeepDive.py 2.5           # Migration Patterns
    python generateDeepDive.py 3.1 --dry-run # Preview topics only
        """,
    )
    parser.add_argument(
        "section",
        help="Section number (e.g., 2.6, 3.1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and display topics without generating",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Max parallel workers (default: {MAX_WORKERS})",
    )

    args = parser.parse_args()

    # Parse section topics and get section name
    logger.info(f"Parsing Section {args.section} from guide...")

    try:
        topics = parse_section_topics(args.section)
        section_name = get_section_name(args.section)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    if not topics:
        logger.error(f"No subsections found in Section {args.section}")
        return 1

    logger.info(f"Section: {section_name}")
    logger.info(f"Found {len(topics)} subsections:")
    for i, topic in enumerate(topics, 1):
        logger.info(f"  {i}. {topic}")

    if args.dry_run:
        logger.info("Dry run complete - no guides generated")
        return 0

    # Check API configuration
    settings = get_settings()
    if not settings.is_fully_configured():
        logger.error("API keys not configured. Set GEMINI_API_KEY in .env")
        return 1

    # Initialize file manager
    file_manager = FileManager()
    available, msg = file_manager.is_cyrus_available()
    if not available:
        logger.error(f"Cyrus not accessible: {msg}")
        return 1

    # Generate guides in parallel
    logger.info(f"Starting parallel generation with {args.workers} workers...")

    results: list[TopicResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_topic = {
            executor.submit(generate_topic, topic, file_manager, i + 1): topic
            for i, topic in enumerate(topics)
        }

        # Collect results as they complete
        for future in as_completed(future_to_topic):
            topic = future_to_topic[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Exception for {topic}: {e}")
                results.append(TopicResult(
                    topic=topic,
                    success=False,
                    filepath="",
                    slug=title_to_slug(topic),
                    error=str(e),
                ))

    # Summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"SUMMARY: {len(successful)}/{len(results)} topics completed successfully")
    logger.info("=" * 60)

    if successful:
        logger.info("\nGenerated files:")
        for r in successful:
            logger.info(f"  - {Path(r.filepath).name}")

    if failed:
        logger.info("\nFailed topics:")
        for r in failed:
            logger.info(f"  - {r.topic}: {r.error}")

    # Update Knowledge Base sidebar with new section
    if successful:
        logger.info("")
        logger.info("Updating Knowledge Base sidebar...")
        successful_slugs = [r.slug for r in successful]
        sidebar_updated = update_knowledge_base_sidebar(section_name, successful_slugs)

        if sidebar_updated:
            logger.info(f"Added '{section_name}' section to sidebar with {len(successful_slugs)} docs")
        else:
            logger.warning("Failed to update sidebar - manual update required")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. cd /Users/omega/Projects/Cyrus")
    logger.info("  2. npm run sync:nebula")
    logger.info("  3. Add Mermaid diagrams to generated docs")
    logger.info("  4. npm run sync:nebula && npm run build")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
