import logging
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Column, Integer, String, Text, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

from app.core.config import config

from app.db.base import Base

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SQLAlchemy Model Definition for Chunks & Embeddings
# -----------------------------------------------------------------------------
class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    chunk_id = Column(String(255), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False, index=True)
    chunk_type = Column(String(50), nullable=False, default="text")
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB, nullable=False, default={})
    embedding = Column(
        Vector(config.gemini.embedding_dimension),
        nullable=True,
    )


# -----------------------------------------------------------------------------
# Database Setup & Table Initialization
# -----------------------------------------------------------------------------
async def init_db_schema(db: AsyncSession) -> None:
    """
    Ensures pgvector extension exists and creates the document_chunks table if missing.
    """
    logger.info("Ensuring 'vector' extension and database schema are initialized...")
    
    # Enable pgvector extension
    await db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    await db.commit()

    # Create tables synchronously via connection
    conn = await db.connection()
    await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully.")


# -----------------------------------------------------------------------------
# Chunk Indexing & Persistence Logic
# -----------------------------------------------------------------------------
async def index_document_chunks(
    db: AsyncSession,
    chunks: List[Dict[str, Any]],
) -> int:
    """
    Inserts or updates document chunks along with their Gemini vector embeddings into pgvector.
    Returns the count of successfully indexed chunks.
    """
    if not chunks:
        logger.warning("No chunks provided to index_document_chunks.")
        return 0

    indexed_count = 0

    # Ensure extension/table exist
    await init_db_schema(db)

    for chunk_data in chunks:
        meta = chunk_data.get("metadata", {})
        chunk_id = meta.get("chunk_id", f"{meta.get('filename')}_{chunk_data.get('chunk_index')}")

        # Check if chunk already exists
        query = select(DocumentChunkModel).where(DocumentChunkModel.chunk_id == chunk_id)
        result = await db.execute(query)
        existing_chunk = result.scalar_one_or_none()

        if existing_chunk:
            # Update existing chunk record
            existing_chunk.content = chunk_data["content"]
            existing_chunk.embedding = chunk_data.get("embedding")
            existing_chunk.metadata_ = meta
            existing_chunk.chunk_type = chunk_data.get("chunk_type", "text")
        else:
            # Create new chunk record
            new_chunk = DocumentChunkModel(
                chunk_id=chunk_id,
                filename=meta.get("filename", "unknown"),
                chunk_type=chunk_data.get("chunk_type", "text"),
                chunk_index=chunk_data.get("chunk_index", 0),
                content=chunk_data["content"],
                metadata_=meta,
                embedding=chunk_data.get("embedding"),
            )
            db.add(new_chunk)

        indexed_count += 1

    await db.commit()
    logger.info("Successfully indexed %d chunks into PostgreSQL/pgvector", indexed_count)
    return indexed_count


async def create_vector_index(db: AsyncSession) -> None:
    """
    Optionally creates an HNSW index on the embedding column for accelerated cosine similarity search.
    """
    logger.info("Creating HNSW index on document_chunks.embedding...")
    index_sql = text("""
        CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
        ON document_chunks USING hnsw (embedding vector_cosine_ops);
    """)
    await db.execute(index_sql)
    await db.commit()
    logger.info("HNSW index successfully created/verified.")