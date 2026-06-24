#!/usr/bin/env python
"""Index earnings transcripts into the RAG vector store."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingest import ingest_all

logging.basicConfig(level=logging.INFO)


def main():
    result = ingest_all()
    transcripts = result.get("transcripts", {})
    print(f"Indexed {transcripts.get('indexed_files', 0)} files")
    print(f"Total chunks: {transcripts.get('total_chunks', 0)}")


if __name__ == "__main__":
    main()
