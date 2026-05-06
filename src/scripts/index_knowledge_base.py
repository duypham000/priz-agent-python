"""Seed the Personal Knowledge Base (pgvector) from local files.

Usage:
    python src/scripts/index_knowledge_base.py --source-dir ./docs
    python src/scripts/index_knowledge_base.py --source-dir ./docs --glob "**/*.md"
    python src/scripts/index_knowledge_base.py --source-dir ./docs --glob "**/*.txt" --chunk-size 1000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


async def index_files(source_dir: str, glob_pattern: str, chunk_size: int) -> None:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    from src.memory.long_term import LongTermMemory
    from src.settings import settings

    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error("Source directory does not exist: %s", source_dir)
        sys.exit(1)

    files = list(source_path.glob(glob_pattern))
    if not files:
        logger.warning("No files matched pattern '%s' in '%s'", glob_pattern, source_dir)
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.gemini_api_key,
    )
    memory = LongTermMemory(postgres_url=settings.postgres_url, embeddings=embeddings)

    total_chunks = 0
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                logger.warning("Skipping empty file: %s", file_path)
                continue
            chunks = _chunk_text(text, chunk_size)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": str(file_path.relative_to(source_path)),
                    "chunk": i,
                    "total_chunks": len(chunks),
                }
                await memory.embed_and_store(chunk, metadata)
                total_chunks += 1
            logger.info("Indexed %s → %d chunk(s)", file_path.name, len(chunks))
        except Exception as exc:
            logger.error("Failed to index %s: %s", file_path, exc)

    logger.info("Done. Total chunks indexed: %d from %d file(s)", total_chunks, len(files))


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into the knowledge base")
    parser.add_argument("--source-dir", required=True, help="Directory containing documents to index")
    parser.add_argument("--glob", default="**/*.md", help="Glob pattern for files (default: **/*.md)")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters (default: 800)")
    args = parser.parse_args()

    asyncio.run(index_files(args.source_dir, args.glob, args.chunk_size))


if __name__ == "__main__":
    main()
