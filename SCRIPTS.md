# ProfessorGemini Scripts

Quick reference for available CLI scripts and the RAG system architecture.

## RAG System Overview

Professor Gemini uses semantic retrieval (RAG) to provide relevant context to Gemini, reducing token usage by ~94% compared to sending the full Knowledge Base.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  AUTHORING (Cyrus)                                              │
│  /Users/omega/Projects/Cyrus/                                   │
│  - gemini-responses/*.md          (Knowledge Base guides)       │
│  - src/data/questions.ts          (Interview Questions)         │
│  - src/data/blindspots.ts         (Deep Dive topics)            │
│  - src/data/knowledge-base-wiki.ts (Wiki Entries)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ DocumentSyncer reads, parses TS, embeds
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  QDRANT CLOUD (Shared Storage)                                  │
│  Collection: professor_gemini                                   │
│  - 768-dim vectors (gemini-embedding-001)                       │
│  - Full document content in payload                             │
│  - Hash-based change detection for incremental sync             │
│  - Reuses IngredientScanner cluster                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Semantic search via QdrantManager
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PROFESSOR GEMINI (Consumer)                                    │
│  - Queries Qdrant for top-5 relevant docs (~150KB)              │
│  - Falls back to full context if RAG fails                      │
│  - Works in cloud deployment (no filesystem dependency)         │
└─────────────────────────────────────────────────────────────────┘
```

### Document Sources

| Source | Type | Doc ID Format |
|--------|------|---------------|
| Knowledge Base | Markdown | `kb:topic-slug` |
| Interview Questions | TypeScript | `questions:question-id` |
| Blindspots | TypeScript | `blindspots:topic-slug` |
| Wiki | TypeScript | `wiki:entry-slug` |

**Total: 400+ documents indexed in Qdrant**

### Token Savings

| Mode | Context Size | Tokens | Cost/Request |
|------|--------------|--------|--------------|
| Full Context | 2.5M chars | ~625K | ~$0.62 |
| RAG (top-5) | 150K chars | ~37K | ~$0.04 |

---

## Scripts

### `app.py` - Streamlit Web UI
Interactive web interface for generating Knowledge Base guides.

```bash
streamlit run app.py --server.port 8502
```

### `syncRag.py` - RAG Document Syncer
Syncs documents from Cyrus to Qdrant with hash-based change detection.

```bash
# Sync all sources (KB, questions, blindspots, wiki)
python syncRag.py sync

# Force full re-index (ignores hashes)
python syncRag.py sync --force

# List all indexed documents
python syncRag.py list
python syncRag.py list --source kb

# Check if sync is needed
python syncRag.py status

# Collection statistics
python syncRag.py stats

# Delete specific document
python syncRag.py delete --doc-id kb:error-budgets

# Purge all docs from a source
python syncRag.py purge --source questions
```

**Features:**
- Hash-based change detection: Only re-indexes modified files
- File mtime staleness check: Fast startup when unchanged
- TypeScript parsing: Extracts data from `.ts` exports on-the-fly
- Orphan cleanup: Removes deleted documents from Qdrant

### `syncNebula.js` - Sync Documents to Cyrus
Syncs markdown files to Cyrus Knowledge Base and Scratch Pad TypeScript data files.

```bash
node syncNebula.js
```

**Sources:**
- `~/Documents/Job Search/LLM Suggestions/` → Scratch Pad
- `Cyrus/gemini-responses/` → Knowledge Base

**Outputs:**
- `Cyrus/src/data/scratch-pad.ts`
- `Cyrus/src/data/knowledge-base.ts`

### `generateDeepDive.py` - Batch Section Generator
Generates Knowledge Base articles for entire sections of the Cyrus System Design guide. Automatically updates the Knowledge Base sidebar.

```bash
# Preview topics without generating
python generateDeepDive.py 2.6 --dry-run

# Generate all subsections for Section 2.6 (Communication Patterns)
python generateDeepDive.py 2.6

# Generate with custom worker count
python generateDeepDive.py 2.5 --workers 3
```

**What it does:**
1. Parses the System Design guide TSX to extract subsection titles
2. Runs `Pipeline.execute()` in parallel (max 5 workers by default)
3. Saves each guide to `gemini-responses/` via `FileManager`
4. Updates the Knowledge Base `page.tsx` with a new sidebar section

**Section examples:**
| Section | Name | Topics |
|---------|------|--------|
| 2.5 | Migration Patterns | Strangler Fig, Branch by Abstraction, Dual-Write, CDC, Deployment Strategies |
| 2.6 | Communication Patterns | REST/gRPC/GraphQL, Queues/Pub-Sub, Polling/WebSockets, Idempotency |

**After running:**
```bash
cd /Users/omega/Projects/Cyrus
npm run sync:nebula
# Add Mermaid diagrams to generated docs
npm run sync:nebula && npm run build
```

---

## Core Modules

| Module | Description |
|--------|-------------|
| `core/pipeline.py` | 4-step Pipeline: Base Knowledge → Topic Split → Deep Dive → Synthesis |
| `core/gemini_client.py` | Gemini API client with grounding and caching |
| `core/bar_raiser.py` | Quality validation and confidence scoring |
| `core/local_processing.py` | Section content processing without API calls |
| `core/qdrant_manager.py` | **NEW** - Qdrant abstraction: embeddings, upsert, search, delete |
| `core/document_syncer.py` | **NEW** - Hash-based sync with TypeScript parsing |
| `core/rag_retriever.py` | **NEW** - Semantic search and context building |
| `core/single_prompt_pipeline.py` | **MODIFIED** - Uses RAG with fallback to full context |
| `core/context_loader.py` | Loads full Knowledge Base (fallback mode) |
| `core/claude_client.py` | Claude API client (optional critique mode) |
| `utils/file_utils.py` | FileManager for saving guides to Cyrus |
| `utils/logging_utils.py` | Structured logging utilities |
| `config/settings.py` | Environment configuration with Qdrant/RAG settings |

---

## Environment Setup

```bash
cd /Users/omega/Projects/ProfessorGemini
source .venv/bin/activate
```

### Required Environment Variables

```bash
# Google Gemini API (required)
GEMINI_API_KEY=your_gemini_api_key

# Qdrant Cloud (required for RAG)
QDRANT_URL=https://xxx.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
```

### Optional Environment Variables

```bash
# Claude API (only if USE_CLAUDE=true)
ANTHROPIC_API_KEY=your_anthropic_api_key
USE_CLAUDE=false

# RAG Configuration
RAG_ENABLED=true          # Enable semantic retrieval (default: true)
RAG_TOP_K=5               # Documents to retrieve (default: 5)

# Cyrus Integration
CYRUS_ROOT_PATH=/Users/omega/Projects/Cyrus

# Pipeline Optimization
ENABLE_CRITIQUE=false     # Skip Bar Raiser loop (faster)
LOCAL_SYNTHESIS=true      # Use local concatenation (no Gemini call)
```

---

## Common Workflows

### Initial Setup
```bash
# 1. Install dependencies
cd /Users/omega/Projects/ProfessorGemini
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Initial RAG sync
python syncRag.py sync
```

### After Editing Cyrus Content
```bash
# Sync changes to Qdrant
python syncRag.py sync

# Verify
python syncRag.py list --source kb
python syncRag.py stats
```

### Generating New Guides
```bash
# 1. Generate via Streamlit UI
streamlit run app.py --server.port 8502

# 2. Sync to Cyrus
cd /Users/omega/Projects/Cyrus
npm run sync:nebula

# 3. Sync to Qdrant for RAG
cd /Users/omega/Projects/ProfessorGemini
python syncRag.py sync
```

---

## Qdrant Collection Schema

```
Collection: professor_gemini
Vector: 768 dimensions (gemini-embedding-001, COSINE distance)

Payload per document:
├── doc_id: str           # "kb:error-budgets", "questions:q-001"
├── source: str           # "kb", "questions", "blindspots", "wiki"
├── title: str            # Document title
├── content: str          # Full document content
├── content_hash: str     # MD5 hash for change detection
├── indexed_at: str       # ISO timestamp
├── char_count: int       # Content length
└── metadata: dict        # Source-specific metadata
```

---

## Troubleshooting

### RAG Not Working
1. Check Qdrant configuration: `python syncRag.py stats`
2. Verify documents indexed: `python syncRag.py list`
3. Check environment: `RAG_ENABLED=true` and Qdrant keys set

### Sync Errors
1. Check Cyrus path exists: `ls $CYRUS_ROOT_PATH`
2. Verify TypeScript files: `ls Cyrus/src/data/*.ts`
3. Run with force: `python syncRag.py sync --force`

### TypeScript Parsing Errors
The parser handles complex TypeScript syntax including:
- Template literals with escaped backticks
- Single quotes (avoiding apostrophe conflicts)
- Unquoted property names
- Trailing commas
- Multi-line comments

If parsing fails, check `core/document_syncer.py` logs for details.
