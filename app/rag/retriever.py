import logging
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.rag.embeddings import get_query_embedding
from app.rag.indexer import DocumentChunkModel

logger = logging.getLogger(__name__)


async def search_similar_chunks(
    db: AsyncSession,
    query: str,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    filename_filter: Optional[str] = None,
    chunk_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Performs cosine vector similarity search against document chunks in pgvector.

    :param db: AsyncSession database connection.
    :param query: Natural language user query.
    :param top_k: Maximum number of chunks to return (defaults to config.rag.top_k).
    :param similarity_threshold: Minimum similarity score cutoff (defaults to config.rag.similarity_threshold).
    :param filename_filter: Optional filename string to filter scope.
    :param chunk_type_filter: Optional filter by 'text' or 'table'.
    :return: List of dicts containing retrieved chunk content, metadata, and similarity scores.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to search_similar_chunks.")
        return []

    # Use defaults from config if not explicitly provided
    effective_top_k = top_k if top_k is not None else config.rag.top_k
    effective_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else config.rag.similarity_threshold
    )

    # 1. Generate query embedding using RETRIEVAL_QUERY task_type
    logger.debug("Generating query embedding for: '%s'", query)
    query_vector = await get_query_embedding(query)

    if not query_vector:
        logger.error("Failed to generate query embedding for search.")
        return []

    # 2. Compute Cosine Distance (1 - cosine_similarity)
    # pgvector provides cosine_distance via .cosine_distance()
    cosine_distance = DocumentChunkModel.embedding.cosine_distance(query_vector)

    # Cosine Similarity = 1 - Cosine Distance
    similarity_score = (1 - cosine_distance).label("similarity_score")

    # 3. Construct SQLAlchemy Query
    stmt = select(DocumentChunkModel, similarity_score).where(
        DocumentChunkModel.embedding.isnot(None)
    )

    # Apply optional metadata/scope filters
    if filename_filter:
        stmt = stmt.where(DocumentChunkModel.filename == filename_filter)

    if chunk_type_filter:
        stmt = stmt.where(DocumentChunkModel.chunk_type == chunk_type_filter)

    # Order by highest similarity first and apply limit
    stmt = stmt.order_by(cosine_distance.asc()).limit(effective_top_k)

    # 4. Execute Query
    logger.debug("Executing pgvector similarity search...")
    result = await db.execute(stmt)
    rows = result.all()

    retrieved_chunks: List[Dict[str, Any]] = []

    for chunk_model, score in rows:
        # Safely extract metadata dict (handling attribute naming differences)
        meta = (
            getattr(chunk_model, "metadata_", None)
            or getattr(chunk_model, "metadata", {})
            or {}
        )

        score_float = float(score) if score is not None else 0.0

        # Filter out results below configured similarity threshold
        if score_float < effective_threshold:
            logger.debug(
                "Skipping chunk %s (score %.3f below threshold %.3f)",
                chunk_model.id,
                score_float,
                effective_threshold,
            )
            continue

        retrieved_chunks.append(
            {
                "chunk_id": chunk_model.id,
                "content": chunk_model.content,
                "filename": meta.get("file_name") or meta.get("filename", "unknown"),
                "page_number": meta.get("page_number", 0),
                "chunk_type": meta.get(
                    "chunk_type", "text"
                ),  # Safely extracted from JSONB
                "metadata": meta,
                "score": float(score),
            }
        )

    logger.info(
        "Retrieved %d relevant chunks for query (top_k=%d, threshold=%.2f)",
        len(retrieved_chunks),
        effective_top_k,
        effective_threshold,
    )

    return retrieved_chunks


def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a single clean markdown context block
    ready for injection into Gemini agent system prompts.
    """
    if not chunks:
        return "No relevant context found in database."

    formatted_sections = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("filename", "Unknown Document")
        page = chunk["metadata"].get("page_number", "N/A")
        chunk_type = chunk.get("chunk_type", "text").upper()
        score = chunk.get("similarity_score", 0.0)

        header = f"--- [Context Chunk {idx}] Source: {source} | Page: {page} | Type: {chunk_type} | Relevancy: {score} ---"
        formatted_sections.append(f"{header}\n{chunk['content']}\n")

    return "\n".join(formatted_sections)
