import json
import logging
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db.models import DocumentChunkModel

logger = logging.getLogger(__name__)


async def insert_chunks_to_pgvector(
    session: AsyncSession,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    objects = [
        DocumentChunkModel(
            content=chunk["content"],
            metadata_=chunk.get(
                "metadata", {}
            ),  # SQLAlchemy handles JSONB dicts directly
            embedding=emb,  # Accepts native list[float]
        )
        for chunk, emb in zip(chunks, embeddings)
    ]

    session.add_all(objects)
    await session.commit()
