"""Batch lexicon generator - 5 terms per KB document.

Processes all Knowledge Base documents in parallel batches of 5,
extracting 5 key lexicon terms from each using Gemini.

Usage:
    cd /Users/omega/Projects/ProfessorGemini
    source .venv/bin/activate
    python scripts/batch_lexicon.py
"""

import asyncio
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gemini_client import GeminiClient
from config.settings import get_settings


# Prompt template for lexicon extraction
LEXICON_PROMPT = """You are a technical glossary expert for Principal TPM interviews at Mag7 companies.

Extract exactly 5 key technical terms from this document that would be valuable for a Principal TPM to know. Focus on:
- Domain-specific terminology unique to this topic
- Concepts that demonstrate Principal-level depth
- Terms that might appear in system design or behavioral interviews

DOCUMENT TITLE: {title}

DOCUMENT CONTENT:
{content}

Return ONLY a valid JSON array with exactly 5 terms. No markdown, no explanation, just the JSON:
[
  {{"term": "Term Name", "definition": "1-2 sentence definition at Principal TPM level"}},
  {{"term": "Another Term", "definition": "Clear, expert-level definition"}}
]"""

# Regex to strip YAML frontmatter
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


@dataclass
class DocumentTerms:
    """Result from processing a single document."""

    filename: str
    title: str
    terms: list[dict]
    success: bool
    error: str = ""


def strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    return FRONTMATTER_PATTERN.sub("", content).strip()


def extract_title(content: str) -> str:
    """Extract title from first H1 heading or first line."""
    lines = content.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    # Fallback to first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()[:100]
    return "Untitled"


def parse_json_terms(response_text: str) -> list[dict]:
    """Parse JSON array of terms from Gemini response.

    Handles common response formats:
    - Clean JSON array
    - JSON wrapped in markdown code blocks
    - JSON with trailing text
    """
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        # Find the actual JSON start
        lines = text.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            if line.startswith("```") and in_json:
                break
            if in_json:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # Find JSON array in text
    start = text.find("[")
    end = text.rfind("]") + 1

    if start == -1 or end == 0:
        return []

    json_str = text[start:end]

    try:
        terms = json.loads(json_str)
        if isinstance(terms, list):
            # Validate structure
            valid_terms = []
            for term in terms:
                if isinstance(term, dict) and "term" in term and "definition" in term:
                    valid_terms.append({
                        "term": str(term["term"]),
                        "definition": str(term["definition"]),
                    })
            return valid_terms[:5]  # Limit to 5
    except json.JSONDecodeError:
        pass

    return []


async def process_single(gemini: GeminiClient, doc_path: Path) -> DocumentTerms:
    """Generate 5 lexicon terms for one document."""
    try:
        content = doc_path.read_text(encoding="utf-8")
        clean_content = strip_frontmatter(content)
        title = extract_title(clean_content)

        # Limit content to avoid token limits (15K chars ~ 4-5K tokens)
        prompt = LEXICON_PROMPT.format(
            title=title,
            content=clean_content[:15000],
        )

        response = await gemini._generate_async(prompt, "lexicon")

        if not response.success:
            return DocumentTerms(
                filename=doc_path.name,
                title=title,
                terms=[],
                success=False,
                error=response.error or "Generation failed",
            )

        terms = parse_json_terms(response.content)

        if not terms:
            return DocumentTerms(
                filename=doc_path.name,
                title=title,
                terms=[],
                success=False,
                error="Failed to parse JSON terms from response",
            )

        return DocumentTerms(
            filename=doc_path.name,
            title=title,
            terms=terms,
            success=True,
        )

    except Exception as e:
        return DocumentTerms(
            filename=doc_path.name,
            title="Error",
            terms=[],
            success=False,
            error=str(e),
        )


async def process_batch(
    gemini: GeminiClient,
    docs: list[Path],
    batch_num: int,
    total_batches: int,
) -> list[DocumentTerms]:
    """Process up to 5 documents in parallel."""
    print(f"  Batch {batch_num}/{total_batches}: Processing {len(docs)} documents...")

    tasks = [process_single(gemini, doc) for doc in docs]
    results = await asyncio.gather(*tasks)

    # Report results
    success_count = sum(1 for r in results if r.success)
    print(f"  Batch {batch_num}/{total_batches}: {success_count}/{len(docs)} succeeded")

    return list(results)


async def main() -> None:
    """Main entry point for batch lexicon generation."""
    print("=" * 60)
    print("Batch Lexicon Generator")
    print("=" * 60)

    settings = get_settings()
    gemini = GeminiClient()

    if not gemini.is_available():
        print("ERROR: Gemini API key not configured")
        sys.exit(1)

    # Get KB documents path from Cyrus project
    kb_path = Path(settings.get_gemini_responses_path())
    print(f"\nKnowledge Base: {kb_path}")

    if not kb_path.exists():
        print(f"ERROR: KB path does not exist: {kb_path}")
        sys.exit(1)

    # Find all markdown files
    docs = sorted(kb_path.glob("*.md"))
    print(f"Found {len(docs)} documents")

    if not docs:
        print("ERROR: No markdown files found")
        sys.exit(1)

    # Process in batches of 5
    batch_size = 5
    total_batches = (len(docs) + batch_size - 1) // batch_size
    all_results: list[DocumentTerms] = []

    print(f"\nProcessing in {total_batches} batches of up to {batch_size}...")
    print("-" * 60)

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        batch_num = i // batch_size + 1

        results = await process_batch(gemini, batch, batch_num, total_batches)
        all_results.extend(results)

    # Summary
    print("-" * 60)
    success_count = sum(1 for r in all_results if r.success)
    total_terms = sum(len(r.terms) for r in all_results)
    print(f"\nResults: {success_count}/{len(all_results)} documents processed")
    print(f"Total terms extracted: {total_terms}")

    # Report failures
    failures = [r for r in all_results if not r.success]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  - {f.filename}: {f.error}")

    # Save results
    output_path = Path(__file__).parent.parent / "lexicon_terms.json"
    output = {
        "total_documents": len(all_results),
        "successful": success_count,
        "total_terms": total_terms,
        "documents": [asdict(r) for r in all_results],
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
