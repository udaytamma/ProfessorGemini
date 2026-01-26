#!/usr/bin/env python
"""Sync documents to Qdrant RAG database.

Usage:
    python syncRag.py sync      # Sync documents (with change detection)
    python syncRag.py list      # List indexed documents
    python syncRag.py status    # Check if sync is needed
    python syncRag.py stats     # Collection statistics
    python syncRag.py delete --doc-id kb:topic-name  # Delete a document
    python syncRag.py purge --source kb              # Purge all docs from source
"""

import sys
from core.document_syncer import main

if __name__ == "__main__":
    main()
