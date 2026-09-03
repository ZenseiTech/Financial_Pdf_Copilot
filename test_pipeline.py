import asyncio
import logging
import os
import sys
from pathlib import Path

from redis.asyncio import Redis
from sqlalchemy import text

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import config
from app.db.session import AsyncSessionLocal
from app.ingestion.chunker import process_pdf_to_chunks
from app.rag.embeddings import (
    get_gemini_embedding,
    get_batch_embeddings,
    get_query_embedding,
)
from app.rag.indexer import insert_chunks_to_pgvector
from app.rag.retriever import search_similar_chunks
from app.agent.loop import generate_agent_response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_pipeline")


async def test_database_connection():
    logger.info("--- Testing Database Connection & pgvector Extension ---")
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT version();"))
        db_version = result.scalar()
        logger.info(f"Connected to PostgreSQL: {db_version}")

        ext_result = await session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        )
        ext = ext_result.scalar()
        assert ext == "vector", "pgvector extension is NOT installed in database!"
        logger.info("✓ PostgreSQL & pgvector verified.")


async def test_redis_connection():
    logger.info("--- Testing Redis Connection ---")
    redis_client = Redis(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password,
        decode_responses=True,
    )
    ping = await redis_client.ping()
    assert ping is True, "Failed to ping Redis server!"

    # Test SET and GET
    test_key = "pipeline_test_key"
    await redis_client.setex(test_key, 60, "redis_ok")
    val = await redis_client.get(test_key)
    assert val == "redis_ok", "Redis key set/get failed!"
    await redis_client.delete(test_key)
    await redis_client.aclose()
    logger.info("✓ Redis connection & cache ops verified.")


async def test_embeddings():
    logger.info("--- Testing Gemini Embeddings API ---")
    model_name = config.gemini.embedding_model
    logger.info(
        f"Using embedding model: {model_name} (dim: {config.gemini.embedding_dimension})"
    )

    test_str = "Q3 financial results indicate revenue growth of 15% year-over-year."
    vec = await get_gemini_embedding(test_str, task_type="RETRIEVAL_DOCUMENT")

    assert (
        len(vec) == config.gemini.embedding_dimension
    ), f"Expected vector dim {config.gemini.embedding_dimension}, got {len(vec)}"
    logger.info(f"✓ Single embedding generated successfully (dim: {len(vec)}).")

    batch_vecs = await get_batch_embeddings(
        [test_str, "Operating expenses remained flat."], batch_size=2
    )
    assert len(batch_vecs) == 2, "Batch embedding count mismatch!"
    logger.info("✓ Batch embeddings generated successfully.")


async def test_ingestion_and_retrieval():
    logger.info("--- Testing PDF Chunker & Vector Search ---")
    sample_pdf = Path("data/sample_report.pdf")

    if not sample_pdf.exists():
        logger.warning(
            f"Sample PDF not found at {sample_pdf}. Creating dummy text chunk for index/retrieval test..."
        )
        chunks = [
            {
                "content": "| Item | Q3 Revenue |\n|---|---|\n| Net Sales | $1,250,000 |",
                "metadata": {
                    "file_name": "test_dummy.pdf",
                    "page_number": 1,
                    "chunk_type": "table",
                },
            }
        ]
    else:
        chunks = process_pdf_to_chunks(str(sample_pdf))
        logger.info(f"Chunker extracted {len(chunks)} chunks locally.")

    assert len(chunks) > 0, "No chunks generated from chunker!"

    # Embed and index test chunks
    contents = [c["content"] for c in chunks]
    embeddings = await get_batch_embeddings(contents)

    async with AsyncSessionLocal() as session:
        await insert_chunks_to_pgvector(session, chunks, embeddings)
        logger.info(f"Indexed {len(chunks)} chunks into pgvector.")

        # Test retriever cosine search
        query = "What was the Q3 net sales or revenue?"
        results = await search_similar_chunks(session, query, top_k=3)
        assert len(results) > 0, "Retriever returned no context results!"
        logger.info(
            f"Retriever found top match (Score: {results[0].get('score', 'N/A')}):"
        )
        logger.info(f"Content snippet: {results[0]['content'][:100]}...")
        logger.info("✓ Ingestion & Similarity Retrieval verified.")


async def test_llm_streaming():
    logger.info("--- Testing Gemini 3.6 Flash Streaming Agent ---")
    prompt = "Summarize the key differences between a 10-K and 10-Q filing in two concise bullet points."

    logger.info("Streaming response from model...")
    tokens = []

    # Pass db session and prompt explicitly using keyword arguments
    async with AsyncSessionLocal() as session:
        async for chunk in generate_agent_response(db=session, user_query=prompt):
            tokens.append(chunk)
            print(chunk, end="", flush=True)

    print("\n")
    full_text = "".join(tokens)
    assert len(full_text) > 0, "LLM streaming generated empty response!"
    logger.info("✓ Streaming response verified successfully.")


async def main():
    logger.info("Starting End-to-End Pipeline Integration Test...")

    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable is not set!")
        sys.exit(1)

    try:
        await test_database_connection()
        await test_redis_connection()
        await test_embeddings()
        await test_ingestion_and_retrieval()
        await test_llm_streaming()
        logger.info("\n🎉 ALL PIPELINE INTEGRATION TESTS PASSED SUCCESSFULLY! 🎉")
    except Exception as exc:
        logger.error(f"\n❌ Pipeline test failed with error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
