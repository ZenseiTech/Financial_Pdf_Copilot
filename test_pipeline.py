import asyncio
import logging
import sys
from pathlib import Path

# Ensure root directory is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent.loop import stream_agent_response
from app.core.config import config, init_logging
from app.db.session import AsyncSessionLocal, async_engine
from app.rag.embeddings import get_gemini_embedding
from app.rag.indexer import index_document_chunks, init_db_schema
from app.rag.retriever import search_similar_chunks

logger = logging.getLogger("test_pipeline")


async def test_database_connection():
    """1. Test database connection and schema initialization."""
    logger.info("=== STEP 1: Testing Database Connection ===")
    async with AsyncSessionLocal() as db:
        await init_db_schema(db)
    logger.info("✅ Database connection and schema check passed.")


async def test_embedding_generation():
    """2. Test vector embedding generation via Google GenAI SDK."""
    logger.info("=== STEP 2: Testing Gemini Embedding Generation ===")
    sample_text = "Q3 net income increased by 14% year-over-year to $4.2 billion."
    embedding = await get_gemini_embedding(sample_text)
    
    assert len(embedding) == config.gemini.embedding_dimension, (
        f"Expected dimension {config.gemini.embedding_dimension}, got {len(embedding)}"
    )
    logger.info("✅ Gemini embedding generated successfully (Dimension: %d).", len(embedding))
    return embedding


async def test_pgvector_indexing(embedding: list):
    """3. Test indexing sample document chunk into pgvector."""
    logger.info("=== STEP 3: Testing Chunk Indexing in pgvector ===")
    test_chunk = {
        "content": "In Q3 2025, operating profit reached $1.8B with a gross margin of 42%.",
        "chunk_type": "text",
        "chunk_index": 0,
        "embedding": embedding,
        "metadata": {
            "filename": "test_q3_report.pdf",
            "page_number": 1,
            "chunk_id": "test_q3_report_0",
        },
    }

    async with AsyncSessionLocal() as db:
        indexed_count = await index_document_chunks(db=db, chunks=[test_chunk])
        assert indexed_count == 1, "Failed to index test chunk."
    logger.info("✅ Sample chunk indexed successfully into pgvector.")


async def test_retrieval():
    """4. Test cosine similarity search against pgvector."""
    logger.info("=== STEP 4: Testing Vector Similarity Search ===")
    query = "What was the operating profit and gross margin in Q3?"
    
    async with AsyncSessionLocal() as db:
        results = await search_similar_chunks(
            db=db,
            query=query,
            filename_filter="test_q3_report.pdf",
            top_k=2,
            similarity_threshold=0.3,
        )
        
        assert len(results) > 0, "Vector search returned no results."
        logger.info("✅ Retrieved %d matching chunk(s). Best match score: %.4f", 
                    len(results), results[0]["similarity_score"])
        logger.info("   Retrieved Content: %s", results[0]["content"])


async def test_agent_chat_stream():
    """5. Test Gemini agent streaming response loop with RAG context."""
    logger.info("=== STEP 5: Testing Agent Streaming Response ===")
    user_query = "Summarize the Q3 2025 financial metrics from the report."
    
    print("\n--- Agent Streamed Response Start ---")
    async with AsyncSessionLocal() as db:
        async for chunk in stream_agent_response(
            db=db,
            user_query=user_query,
            filename_filter="test_q3_report.pdf",
        ):
            print(chunk, end="", flush=True)
    print("\n--- Agent Streamed Response End ---\n")
    logger.info("✅ Agent response stream completed successfully.")


async def main():
    init_logging()
    logger.info("Starting End-to-End System Verification...")

    try:
        await test_database_connection()
        embedding = await test_embedding_generation()
        await test_pgvector_indexing(embedding)
        await test_retrieval()
        await test_agent_chat_stream()
        
        logger.info("🎉 ALL SYSTEM TESTS PASSED SUCCESSFULLY!")

    except Exception as exc:
        logger.critical("❌ Pipeline verification failed: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())