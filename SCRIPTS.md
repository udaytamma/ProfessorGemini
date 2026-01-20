# ProfessorGemini Scripts

Quick reference for available CLI scripts in this project.

## Scripts

### `app.py` - Streamlit Web UI
Interactive web interface for generating Knowledge Base guides.

```bash
streamlit run app.py --server.port 8502
```

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

## Core Modules

| Module | Description |
|--------|-------------|
| `core/pipeline.py` | 4-step Pipeline: Base Knowledge → Topic Split → Deep Dive → Synthesis |
| `core/gemini_client.py` | Gemini API client with grounding and caching |
| `core/bar_raiser.py` | Quality validation and confidence scoring |
| `core/local_processing.py` | Section content processing without API calls |
| `utils/file_utils.py` | FileManager for saving guides to Cyrus |
| `utils/logging_utils.py` | Structured logging utilities |
| `config/settings.py` | Environment configuration |

## Environment Setup

```bash
cd /Users/omega/Projects/ProfessorGemini
source .venv/bin/activate
```

Required env vars in `.env`:
- `GEMINI_API_KEY` - Google Gemini API key
- `ANTHROPIC_API_KEY` - (Optional) For Claude-based quality checks
