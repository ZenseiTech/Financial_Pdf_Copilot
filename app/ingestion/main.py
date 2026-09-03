import argparse
import asyncio
import logging
from pathlib import Path

from app.db.session import AsyncSessionLocal, init_db
from app.ingestion.chunker import process_pdf_to_chunks
from app.rag.embeddings import get_batch_embeddings
from app.rag.indexer import insert_chunks_to_pgvector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_document(pdf_path: str):
    """
    Orchestrates the local PDF ingestion pipeline:
    1. Local chunking via chunker.py (Zero Gemini token cost)
    2. Batch vector embedding generation
    3. Storage in PostgreSQL / pgvector
    """
    # Ensure tables exist before running insertion queries
    await init_db()

    path = Path(pdf_path)
    if not path.exists():
        logger.error("File not found at path: %s", pdf_path)
        return

    # 1. Extract chunks locally using app.ingestion.chunker
    logger.info("Extracting chunks from %s...", path.name)
    chunks = process_pdf_to_chunks(str(path))

    if not chunks:
        logger.warning("No valid content extracted from %s", pdf_path)
        return

    logger.info("Generated %d local chunks from %s", len(chunks), path.name)

    # 2. Extract raw text content for batch embedding
    contents = [c["content"] for c in chunks]

    # 3. Generate embeddings via Gemini API
    logger.info("Generating embeddings for %d chunks...", len(contents))
    embeddings = await get_batch_embeddings(contents, task_type="RETRIEVAL_DOCUMENT")

    # Filter out any chunks where embedding failed
    valid_chunks = []
    valid_embeddings = []
    for chunk, emb in zip(chunks, embeddings):
        if emb:
            valid_chunks.append(chunk)
            valid_embeddings.append(emb)

    # 4. Insert vectors and metadata into pgvector
    async with AsyncSessionLocal() as session:
        logger.info("Indexing %d chunks into pgvector...", len(valid_chunks))
        await insert_chunks_to_pgvector(session, valid_chunks, valid_embeddings)

    logger.info("Successfully ingested %s!", path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF documents into pgvector.")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF file")
    args = parser.parse_args()

    asyncio.run(ingest_document(args.file))
