import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import config, init_logging
from app.db.session import AsyncSessionLocal
from app.ingestion.chunker import chunk_financial_document
from app.ingestion.pdf_parser import extract_pdf_content
from app.rag.embeddings import get_gemini_embedding
from app.rag.indexer import index_document_chunks

# Module-level logger scoped to app.ingestion namespace
logger = logging.getLogger(__name__)


async def process_financial_pdf(file_path: Path) -> Dict[str, Any]:
    """
    Ingestion pipeline for a single financial PDF document:
    1. Extracts text and financial tables.
    2. Chunks content with section and table boundary awareness.
    3. Generates vector embeddings using Gemini API.
    4. Indexes chunks and vectors in PostgreSQL (pgvector).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Financial PDF not found: {file_path}")

    logger.info("Starting ingestion pipeline for %s", file_path.name)

    # 1. Parse text and financial tables from PDF
    logger.debug("Extracting PDF contents for %s...", file_path.name)
    parsed_doc = await extract_pdf_content(file_path)
    logger.info(
        "Extracted %d pages and %d tables from %s",
        parsed_doc.get("page_count", 0),
        len(parsed_doc.get("tables", [])),
        file_path.name,
    )

    # 2. Chunk text and table data while retaining financial context
    logger.debug("Chunking document content...")
    chunks: List[Dict[str, Any]] = chunk_financial_document(
        text=parsed_doc.get("text", ""),
        tables=parsed_doc.get("tables", []),
        metadata={
            "filename": file_path.name,
            "path": str(file_path.resolve()),
        },
    )
    logger.info("Generated %d chunks from %s", len(chunks), file_path.name)

    # 3. Generate Gemini vector embeddings
    logger.debug("Generating embeddings via Gemini SDK...")
    for chunk in chunks:
        vector = await get_gemini_embedding(chunk["content"])
        chunk["embedding"] = vector

    # 4. Store chunks & embeddings in PostgreSQL via AsyncSession
    logger.debug("Persisting chunks to vector database...")
    async with AsyncSessionLocal() as db:
        indexed_count = await index_document_chunks(db=db, chunks=chunks)

    logger.info(
        "Successfully ingested %s (%d chunks indexed)",
        file_path.name,
        indexed_count,
    )
    return {
        "status": "success",
        "filename": file_path.name,
        "chunks_indexed": indexed_count,
    }


async def process_directory(directory_path: Path):
    """Batch-processes all PDF files inside a directory."""
    dir_path = Path(directory_path)
    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in directory: %s", dir_path)
        return

    logger.info("Found %d PDFs in %s for batch ingestion", len(pdf_files), dir_path)
    
    for pdf_file in pdf_files:
        try:
            await process_financial_pdf(pdf_file)
        except Exception as err:
            logger.error("Failed to process %s: %s", pdf_file.name, err, exc_info=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI utility for processing and indexing financial PDFs."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Path to a single financial PDF file.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        help="Path to a directory containing PDF files for batch processing.",
    )
    return parser.parse_args()


async def async_main():
    init_logging()
    args = parse_args()

    if args.file:
        await process_financial_pdf(args.file)
    elif args.dir:
        await process_directory(args.dir)
    else:
        logger.error("Please supply a file (--file) or directory (--dir) to ingest.")
        sys.exit(1)


def main():
    """CLI execution entrypoint."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Ingestion process aborted by user.")
    except Exception as exc:
        logger.critical("Ingestion pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()