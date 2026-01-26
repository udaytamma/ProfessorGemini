"""Document syncer with hash-based change detection.

Syncs all Nebula content from Cyrus to Qdrant:
- Knowledge Base (markdown files)
- Scratch Pad (markdown files)
- Interview Questions (TypeScript)
- Blindspots (TypeScript)
- Wiki (TypeScript)

Features:
- Detects changes via MD5 hash
- Only re-indexes modified files
- Removes orphaned documents
- Parses TypeScript data files on-the-fly

Designed to run on Professor Gemini startup.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config.settings import get_settings
from core.qdrant_manager import QdrantDocument, QdrantManager

logger = logging.getLogger(__name__)


# =============================================================================
# TypeScript Parsing Utilities
# =============================================================================


def parse_typescript_array(ts_content: str, array_name: str) -> list[dict]:
    """Parse a TypeScript array export into Python dicts.

    Extracts the array content between 'export const {array_name} = [' and '];'
    then parses it as JSON (with some TS-specific cleanup).

    Args:
        ts_content: Full TypeScript file content.
        array_name: Name of the exported array (e.g., 'questions').

    Returns:
        List of dictionaries parsed from the array.
    """
    # Find the array start pattern
    pattern = rf"export const {array_name}[^=]*=\s*\["
    match = re.search(pattern, ts_content)
    if not match:
        logger.warning(f"Could not find array '{array_name}' in TypeScript content")
        return []

    # Find the matching closing bracket
    start_idx = match.end() - 1  # Include the opening bracket
    bracket_count = 0
    end_idx = start_idx

    for i, char in enumerate(ts_content[start_idx:], start=start_idx):
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i + 1
                break

    array_content = ts_content[start_idx:end_idx]

    # Clean up TypeScript-specific syntax for JSON parsing
    # Strategy: Handle template literals separately, then convert regular strings

    # 1. Remove multi-line comments
    array_content = re.sub(r"/\*[\s\S]*?\*/", "", array_content)

    # 2. Remove single-line comments (only at line start or after whitespace)
    #    Avoids removing // inside URLs like https://...
    array_content = re.sub(r"(^|\s)//[^\n]*", r"\1", array_content, flags=re.MULTILINE)

    # 3. Handle template literals FIRST (before any quote manipulation)
    #    Replace backtick strings with a placeholder, then restore after other processing
    template_literals: list[str] = []

    def save_template(m: re.Match) -> str:
        """Save template literal and return placeholder."""
        idx = len(template_literals)
        content = m.group(1)
        # Unescape backticks that were escaped in TypeScript: \` -> `
        content = content.replace("\\`", "`")
        # json.dumps handles newlines, quotes, and special chars
        template_literals.append(json.dumps(content))
        return f"__TPL_{idx}__"

    # Match template literals, handling escaped backticks inside: `text with \`nested\` backticks`
    # Pattern: backtick, then (non-backtick/non-backslash OR any escaped char)*, then backtick
    array_content = re.sub(r"`((?:[^`\\]|\\.)*)`", save_template, array_content)

    # 4. Convert single quotes to double quotes
    #    Must avoid matching apostrophes in contractions (don't, it's, What's)
    #    Strategy: Only match single quotes that appear at "string boundaries"
    #    i.e., preceded by: start, whitespace, colon, comma, bracket, paren
    def convert_single_quoted(m: re.Match) -> str:
        """Convert single-quoted string to double-quoted."""
        prefix = m.group(1)  # The character before the quote
        content = m.group(2)
        # Escape any unescaped double quotes inside
        content = content.replace('\\"', "__ESCAPED_DQ__")
        content = content.replace('"', '\\"')
        content = content.replace("__ESCAPED_DQ__", '\\"')
        # Unescape single quotes (no longer needed in double-quoted string)
        content = content.replace("\\'", "'")
        return f'{prefix}"{content}"'

    # Pattern: (boundary char)'(string content)'
    # Boundary: start of string, whitespace, or punctuation that precedes strings
    array_content = re.sub(
        r"(^|[\s:,\[\]{}(])'((?:[^'\\]|\\.)*)'",
        convert_single_quoted,
        array_content,
        flags=re.MULTILINE,
    )

    # 5. Quote unquoted property names: {key: value} → {"key": value}
    #    Do BEFORE restoring templates (placeholders aren't affected)
    array_content = re.sub(
        r'([{\[,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)',
        r'\1"\2"\3',
        array_content,
    )

    # 6. Remove trailing commas before ] or }
    #    Do BEFORE restoring templates
    array_content = re.sub(r",(\s*[}\]])", r"\1", array_content)

    # 7. Restore template literals (already JSON-encoded)
    #    Done LAST so template content isn't affected by prior transformations
    for idx, literal in enumerate(template_literals):
        array_content = array_content.replace(f"__TPL_{idx}__", literal)

    try:
        return json.loads(array_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse TypeScript array '{array_name}': {e}")
        # Log a snippet for debugging
        logger.debug(f"Content snippet: {array_content[:500]}...")
        return []


@dataclass
class SyncResult:
    """Result of a sync operation.

    Attributes:
        total_files: Number of files found in source.
        indexed: Number of files newly indexed.
        skipped: Number of files skipped (unchanged).
        deleted: Number of orphaned documents deleted.
        errors: Number of errors encountered.
        duration_ms: Total sync duration in milliseconds.
    """

    total_files: int
    indexed: int
    skipped: int
    deleted: int
    errors: int
    duration_ms: int


class DocumentSyncer:
    """Syncs documents from filesystem to Qdrant.

    Reads markdown files from configured source directories,
    computes content hashes, and only re-indexes changed files.
    """

    def __init__(self) -> None:
        """Initialize syncer with dependencies."""
        self._settings = get_settings()
        self._qdrant = QdrantManager()

    def _compute_hash(self, content: str) -> str:
        """Compute MD5 hash of content.

        Args:
            content: String content to hash.

        Returns:
            MD5 hex digest.
        """
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_title(self, content: str, filename: str) -> str:
        """Extract title from markdown content.

        Looks for first H1 heading, falls back to filename.

        Args:
            content: Markdown content.
            filename: File name as fallback.

        Returns:
            Extracted or generated title.
        """
        lines = content.strip().split("\n")
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
        # Fallback to filename
        return filename.replace("-", " ").replace(".md", "").title()

    def _extract_metadata(self, content: str, source: str) -> dict:
        """Extract source-specific metadata from content.

        Args:
            content: Document content.
            source: Source identifier.

        Returns:
            Dict of extracted metadata.
        """
        metadata: dict = {}

        # Extract date if present (various formats)
        for line in content.split("\n")[:30]:
            line_lower = line.lower()
            if "date:" in line_lower or "**date:**" in line_lower:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    date_str = parts[1].strip().strip("*")
                    metadata["date"] = date_str
                break

        return metadata

    def sync_source(
        self, source: str, source_path: str, pattern: str = "*.md"
    ) -> SyncResult:
        """Sync all documents from a source directory.

        Args:
            source: Source identifier (e.g., "kb", "scratch").
            source_path: Path to source directory.
            pattern: Glob pattern for files.

        Returns:
            SyncResult with statistics.
        """
        start_time = time.time()

        path = Path(source_path).expanduser()
        if not path.exists():
            logger.warning(f"Source path does not exist: {source_path}")
            return SyncResult(0, 0, 0, 0, 1, 0)

        # Get currently indexed docs for this source
        try:
            indexed_docs = {
                d["doc_id"]: d["content_hash"]
                for d in self._qdrant.list_documents(source_filter=source)
            }
        except Exception as e:
            logger.error(f"Failed to list indexed documents: {e}")
            indexed_docs = {}

        # Track which docs we see in filesystem
        seen_doc_ids: set[str] = set()
        indexed = 0
        skipped = 0
        errors = 0

        # Process all files
        files = list(path.glob(pattern))
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                slug = file_path.stem  # filename without extension
                doc_id = f"{source}:{slug}"
                seen_doc_ids.add(doc_id)

                # Check if content changed
                content_hash = self._compute_hash(content)
                if doc_id in indexed_docs and indexed_docs[doc_id] == content_hash:
                    skipped += 1
                    continue

                # Index the document
                title = self._extract_title(content, file_path.name)
                metadata = self._extract_metadata(content, source)

                doc = QdrantDocument(
                    doc_id=doc_id,
                    source=source,
                    title=title,
                    content=content,
                    content_hash=content_hash,
                    indexed_at=datetime.now(timezone.utc).isoformat(),
                    char_count=len(content),
                    metadata=metadata,
                )

                self._qdrant.upsert(doc)
                indexed += 1
                logger.info(f"Indexed: {doc_id} ({len(content):,} chars)")

            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                errors += 1

        # Delete orphaned documents (in Qdrant but not in filesystem)
        deleted = 0
        orphaned = set(indexed_docs.keys()) - seen_doc_ids
        for doc_id in orphaned:
            try:
                self._qdrant.delete(doc_id)
                deleted += 1
                logger.info(f"Deleted orphan: {doc_id}")
            except Exception as e:
                logger.error(f"Error deleting {doc_id}: {e}")
                errors += 1

        duration_ms = int((time.time() - start_time) * 1000)

        return SyncResult(
            total_files=len(files),
            indexed=indexed,
            skipped=skipped,
            deleted=deleted,
            errors=errors,
            duration_ms=duration_ms,
        )

    def sync_typescript_source(
        self,
        source: str,
        file_path: str,
        array_name: str,
        transform_fn: callable,
    ) -> SyncResult:
        """Sync documents from a TypeScript data file.

        Args:
            source: Source identifier (e.g., "questions", "blindspots").
            file_path: Path to the TypeScript file.
            array_name: Name of the exported array.
            transform_fn: Function to transform each item into (doc_id, title, content).

        Returns:
            SyncResult with statistics.
        """
        start_time = time.time()

        path = Path(file_path).expanduser()
        if not path.exists():
            logger.warning(f"TypeScript file does not exist: {file_path}")
            return SyncResult(0, 0, 0, 0, 1, 0)

        # Get currently indexed docs for this source
        try:
            indexed_docs = {
                d["doc_id"]: d["content_hash"]
                for d in self._qdrant.list_documents(source_filter=source)
            }
        except Exception as e:
            logger.error(f"Failed to list indexed documents: {e}")
            indexed_docs = {}

        # Parse TypeScript file
        ts_content = path.read_text(encoding="utf-8")
        items = parse_typescript_array(ts_content, array_name)

        if not items:
            logger.warning(f"No items found in {file_path}")
            return SyncResult(0, 0, 0, 0, 0, int((time.time() - start_time) * 1000))

        # Track which docs we see
        seen_doc_ids: set[str] = set()
        indexed = 0
        skipped = 0
        errors = 0

        for item in items:
            try:
                doc_id, title, content = transform_fn(item, source)
                seen_doc_ids.add(doc_id)

                # Check if content changed
                content_hash = self._compute_hash(content)
                if doc_id in indexed_docs and indexed_docs[doc_id] == content_hash:
                    skipped += 1
                    continue

                # Index the document
                doc = QdrantDocument(
                    doc_id=doc_id,
                    source=source,
                    title=title,
                    content=content,
                    content_hash=content_hash,
                    indexed_at=datetime.now(timezone.utc).isoformat(),
                    char_count=len(content),
                    metadata={"item_id": item.get("id", "")},
                )

                self._qdrant.upsert(doc)
                indexed += 1
                logger.info(f"Indexed: {doc_id}")

            except Exception as e:
                logger.error(f"Error indexing item from {source}: {e}")
                errors += 1

        # Delete orphaned documents
        deleted = 0
        orphaned = set(indexed_docs.keys()) - seen_doc_ids
        for doc_id in orphaned:
            try:
                self._qdrant.delete(doc_id)
                deleted += 1
                logger.info(f"Deleted orphan: {doc_id}")
            except Exception as e:
                logger.error(f"Error deleting {doc_id}: {e}")
                errors += 1

        duration_ms = int((time.time() - start_time) * 1000)

        return SyncResult(
            total_files=len(items),
            indexed=indexed,
            skipped=skipped,
            deleted=deleted,
            errors=errors,
            duration_ms=duration_ms,
        )

    def _transform_question(self, item: dict, source: str) -> tuple[str, str, str]:
        """Transform a question item into (doc_id, title, content)."""
        item_id = item.get("id", "unknown")
        question = item.get("question", "")
        answer = item.get("answer", "")
        level = item.get("level", "")
        topics = item.get("topics", [])

        doc_id = f"{source}:{item_id}"
        title = question[:80] + "..." if len(question) > 80 else question

        content = f"""# Interview Question

**Level:** {level}
**Topics:** {", ".join(topics)}

## Question

{question}

## Answer

{answer}
"""
        return doc_id, title, content

    def _transform_blindspot(self, item: dict, source: str) -> tuple[str, str, str]:
        """Transform a blindspot item into (doc_id, title, content)."""
        item_id = item.get("id", "unknown")
        question = item.get("question", "")
        answer = item.get("answer", "")
        category = item.get("category", "")
        difficulty = item.get("difficulty", "")
        mastery = item.get("masteryLevel", "")
        why_asked = item.get("whyAsked", "")
        follow_ups = item.get("followUps", [])
        red_flags = item.get("redFlags", [])

        doc_id = f"{source}:{item_id}"
        title = question[:80] + "..." if len(question) > 80 else question

        follow_ups_text = "\n".join(f"- {f}" for f in follow_ups) if follow_ups else "None"
        red_flags_text = "\n".join(f"- {f}" for f in red_flags) if red_flags else "None"

        content = f"""# Blindspot Question

**Category:** {category}
**Difficulty:** {difficulty}
**Mastery Level:** {mastery}

## Question

{question}

## Why This Is Asked

{why_asked}

## Answer

{answer}

## Follow-up Questions

{follow_ups_text}

## Red Flags (Poor Answers)

{red_flags_text}
"""
        return doc_id, title, content

    def _transform_wiki_entry(
        self, entry: dict, provider: str, group: str, source: str
    ) -> tuple[str, str, str]:
        """Transform a wiki entry into (doc_id, title, content)."""
        tool = entry.get("tool", "unknown")
        summary = entry.get("summary", "")
        mag7 = entry.get("mag7", "")
        adoption = entry.get("adoption", "")
        decision = entry.get("decision", "")
        cost_tier = entry.get("costTier", "")

        # Create a slug from provider and tool
        slug = f"{provider.lower()}-{tool.lower()}".replace(" ", "-")
        doc_id = f"{source}:{slug}"
        title = f"{provider} {tool}"

        content = f"""# {provider} {tool}

**Category:** {group}
**Adoption:** {adoption}
**Cost Tier:** {cost_tier or "N/A"}

## Summary

{summary}

## Mag7 Context

{mag7}

## Decision Guidance

{decision or "No specific guidance provided."}
"""
        return doc_id, title, content

    def sync_wiki(self) -> SyncResult:
        """Sync wiki entries from knowledge-base-wiki.ts.

        The wiki has a nested structure that needs special handling.
        """
        start_time = time.time()
        source = "wiki"

        cyrus_path = Path(self._settings.cyrus_root_path)
        wiki_path = cyrus_path / "src" / "data" / "knowledge-base-wiki.ts"

        if not wiki_path.exists():
            logger.warning(f"Wiki file does not exist: {wiki_path}")
            return SyncResult(0, 0, 0, 0, 1, 0)

        # Get currently indexed docs
        try:
            indexed_docs = {
                d["doc_id"]: d["content_hash"]
                for d in self._qdrant.list_documents(source_filter=source)
            }
        except Exception as e:
            logger.error(f"Failed to list indexed documents: {e}")
            indexed_docs = {}

        # Parse wiki sections
        ts_content = wiki_path.read_text(encoding="utf-8")
        sections = parse_typescript_array(ts_content, "knowledgeBaseWikiSections")

        seen_doc_ids: set[str] = set()
        indexed = 0
        skipped = 0
        errors = 0
        total_entries = 0

        for section in sections:
            provider = section.get("provider", "Unknown")
            for group in section.get("groups", []):
                group_name = group.get("name", "Unknown")
                for entry in group.get("entries", []):
                    total_entries += 1
                    try:
                        doc_id, title, content = self._transform_wiki_entry(
                            entry, provider, group_name, source
                        )
                        seen_doc_ids.add(doc_id)

                        content_hash = self._compute_hash(content)
                        if doc_id in indexed_docs and indexed_docs[doc_id] == content_hash:
                            skipped += 1
                            continue

                        doc = QdrantDocument(
                            doc_id=doc_id,
                            source=source,
                            title=title,
                            content=content,
                            content_hash=content_hash,
                            indexed_at=datetime.now(timezone.utc).isoformat(),
                            char_count=len(content),
                            metadata={"provider": provider, "group": group_name},
                        )

                        self._qdrant.upsert(doc)
                        indexed += 1
                        logger.info(f"Indexed: {doc_id}")

                    except Exception as e:
                        logger.error(f"Error indexing wiki entry: {e}")
                        errors += 1

        # Delete orphaned documents
        deleted = 0
        orphaned = set(indexed_docs.keys()) - seen_doc_ids
        for doc_id in orphaned:
            try:
                self._qdrant.delete(doc_id)
                deleted += 1
                logger.info(f"Deleted orphan: {doc_id}")
            except Exception as e:
                logger.error(f"Error deleting {doc_id}: {e}")
                errors += 1

        duration_ms = int((time.time() - start_time) * 1000)

        return SyncResult(
            total_files=total_entries,
            indexed=indexed,
            skipped=skipped,
            deleted=deleted,
            errors=errors,
            duration_ms=duration_ms,
        )

    def sync_all(self) -> dict[str, SyncResult]:
        """Sync all Nebula sources.

        Sources:
        - kb: Knowledge Base markdown files (gemini-responses/)
        - scratch: Scratch Pad markdown files (LLM Suggestions/)
        - questions: Interview questions (questions.ts)
        - blindspots: Blindspot questions (blindspots.ts)
        - wiki: Cloud services wiki (knowledge-base-wiki.ts)

        Returns:
            Dict mapping source name to SyncResult.
        """
        results = {}
        cyrus_path = Path(self._settings.cyrus_root_path)

        # 1. Sync Knowledge Base (markdown)
        kb_path = self._settings.get_gemini_responses_path()
        logger.info(f"Syncing KB from: {kb_path}")
        results["kb"] = self.sync_source(
            source="kb",
            source_path=kb_path,
            pattern="*.md",
        )

        # 2. Sync Scratch Pad (markdown)
        scratch_path = Path("~/Documents/Job Search/LLM Suggestions").expanduser()
        if scratch_path.exists():
            logger.info(f"Syncing Scratch Pad from: {scratch_path}")
            results["scratch"] = self.sync_source(
                source="scratch",
                source_path=str(scratch_path),
                pattern="*.md",
            )
        else:
            logger.info("Scratch Pad path not found, skipping")

        # 3. Sync Interview Questions (TypeScript)
        questions_path = cyrus_path / "src" / "data" / "questions.ts"
        if questions_path.exists():
            logger.info(f"Syncing Questions from: {questions_path}")
            results["questions"] = self.sync_typescript_source(
                source="questions",
                file_path=str(questions_path),
                array_name="questions",
                transform_fn=self._transform_question,
            )
        else:
            logger.info("Questions file not found, skipping")

        # 4. Sync Blindspots (TypeScript)
        blindspots_path = cyrus_path / "src" / "data" / "blindspots.ts"
        if blindspots_path.exists():
            logger.info(f"Syncing Blindspots from: {blindspots_path}")
            results["blindspots"] = self.sync_typescript_source(
                source="blindspots",
                file_path=str(blindspots_path),
                array_name="blindspotQuestions",
                transform_fn=self._transform_blindspot,
            )
        else:
            logger.info("Blindspots file not found, skipping")

        # 5. Sync Wiki (TypeScript - nested structure)
        wiki_path = cyrus_path / "src" / "data" / "knowledge-base-wiki.ts"
        if wiki_path.exists():
            logger.info(f"Syncing Wiki from: {wiki_path}")
            results["wiki"] = self.sync_wiki()
        else:
            logger.info("Wiki file not found, skipping")

        return results

    def is_stale(self) -> bool:
        """Check if any source files have changed since last sync.

        Quick check using file modification times vs indexed_at.
        Checks all Nebula sources: KB, Scratch Pad, Questions, Blindspots, Wiki.

        Returns:
            True if sync is needed, False if up-to-date.
        """
        try:
            # Get most recent indexed_at from Qdrant
            docs = self._qdrant.list_documents()
            if not docs:
                logger.info("No documents indexed yet, sync needed")
                return True

            # Find latest indexed timestamp
            timestamps = [d.get("indexed_at", "") for d in docs if d.get("indexed_at")]
            if not timestamps:
                return True

            latest_indexed = max(timestamps)

            # Parse the timestamp (handles both Z and +00:00 formats)
            if latest_indexed.endswith("Z"):
                latest_indexed = latest_indexed[:-1] + "+00:00"
            indexed_time = datetime.fromisoformat(latest_indexed)

            cyrus_path = Path(self._settings.cyrus_root_path)

            # Check Knowledge Base markdown files
            kb_path = Path(self._settings.get_gemini_responses_path()).expanduser()
            if kb_path.exists():
                for file_path in kb_path.glob("*.md"):
                    file_mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=timezone.utc
                    )
                    if file_mtime > indexed_time:
                        logger.info(f"Stale: {file_path.name} modified after last sync")
                        return True

            # Check Scratch Pad markdown files
            scratch_path = Path("~/Documents/Job Search/LLM Suggestions").expanduser()
            if scratch_path.exists():
                for file_path in scratch_path.glob("*.md"):
                    file_mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=timezone.utc
                    )
                    if file_mtime > indexed_time:
                        logger.info(f"Stale: {file_path.name} modified after last sync")
                        return True

            # Check TypeScript data files
            ts_files = [
                cyrus_path / "src" / "data" / "questions.ts",
                cyrus_path / "src" / "data" / "blindspots.ts",
                cyrus_path / "src" / "data" / "knowledge-base-wiki.ts",
            ]

            for ts_file in ts_files:
                if ts_file.exists():
                    file_mtime = datetime.fromtimestamp(
                        ts_file.stat().st_mtime, tz=timezone.utc
                    )
                    if file_mtime > indexed_time:
                        logger.info(f"Stale: {ts_file.name} modified after last sync")
                        return True

            logger.info("Documents up-to-date")
            return False

        except Exception as e:
            logger.warning(f"Error checking staleness: {e}")
            return True  # Sync to be safe


def sync_if_needed() -> dict[str, SyncResult] | None:
    """Sync documents only if source files have changed.

    Designed to be called on Professor Gemini startup.

    Returns:
        SyncResult dict if sync was performed, None if skipped.
    """
    syncer = DocumentSyncer()

    if syncer.is_stale():
        logger.info("Documents are stale, syncing...")
        return syncer.sync_all()
    else:
        logger.info("Documents up-to-date, skipping sync")
        return None


def main() -> None:
    """CLI entrypoint for manual sync operations."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Sync documents to Qdrant RAG database",
        prog="syncRag",
    )
    parser.add_argument(
        "command",
        choices=["sync", "list", "delete", "purge", "status", "stats"],
        help="Command to execute",
    )
    parser.add_argument("--source", help="Filter by source (kb, scratch, etc)")
    parser.add_argument("--force", action="store_true", help="Force full re-index")
    parser.add_argument("--doc-id", help="Document ID for delete command")
    args = parser.parse_args()

    syncer = DocumentSyncer()
    qdrant = QdrantManager()

    if args.command == "sync":
        if args.force:
            # For force sync, we just run sync_all which will re-check hashes
            # To truly force re-index, we'd need to clear the collection first
            logger.info("Running sync (force flag set, will re-check all files)...")
        results = syncer.sync_all()
        for source, result in results.items():
            print(f"\n{source}:")
            print(f"  Files: {result.total_files}")
            print(f"  Indexed: {result.indexed}")
            print(f"  Skipped: {result.skipped}")
            print(f"  Deleted: {result.deleted}")
            print(f"  Errors: {result.errors}")
            print(f"  Duration: {result.duration_ms}ms")

    elif args.command == "list":
        docs = qdrant.list_documents(source_filter=args.source)
        print(f"\nIndexed documents: {len(docs)}")
        for doc in sorted(docs, key=lambda d: d.get("doc_id", "")):
            print(
                f"  {doc['doc_id']}: {doc.get('title', 'Untitled')} "
                f"({doc.get('char_count', 0):,} chars)"
            )

    elif args.command == "delete":
        if not args.doc_id:
            print("Error: --doc-id required for delete")
            sys.exit(1)
        qdrant.delete(args.doc_id)
        print(f"Deleted: {args.doc_id}")

    elif args.command == "purge":
        if not args.source:
            print("Error: --source required for purge")
            sys.exit(1)
        docs = qdrant.list_documents(source_filter=args.source)
        for doc in docs:
            qdrant.delete(doc["doc_id"])
        print(f"Purged {len(docs)} documents from source: {args.source}")

    elif args.command == "status":
        if syncer.is_stale():
            print("Status: STALE (sync needed)")
        else:
            print("Status: UP-TO-DATE")

    elif args.command == "stats":
        stats = qdrant.get_collection_stats()
        print(f"\nCollection: {qdrant.collection_name}")
        print(f"  Points: {stats.get('points_count', 0)}")
        print(f"  Status: {stats.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
