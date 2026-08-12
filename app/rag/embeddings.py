import asyncio
import logging
from typing import List, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.core.config import config

logger = logging.getLogger(__name__)

# Initialize Google GenAI client
# Expects GEMINI_API_KEY environment variable to be set
client = genai.Client()


async def get_gemini_embedding(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
    max_retries: int = 3,
) -> List[float]:
    """
    Generates a vector embedding for a single text chunk using the Google GenAI SDK.
    Includes exponential backoff retries for transient API errors.
    
    :param text: Text string or Markdown table content to embed.
    :param task_type: Gemini embedding task type ('RETRIEVAL_DOCUMENT' or 'RETRIEVAL_QUERY').
    :param max_retries: Maximum retry attempts on rate limit/transient failures.
    :return: List of floats representing the embedding vector.
    """
    if not text or not text.strip():
        logger.warning("Empty text string provided for embedding generation.")
        return []

    model_name = config.gemini.embedding_model
    target_dim = config.gemini.embedding_dimension

    for attempt in range(1, max_retries + 1):
        try:
            # Execute embedding call via async client
            response = await client.aio.models.embed_content(
                model=model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=target_dim,
                ),
            )

            if response.embedding and response.embedding.values:
                return response.embedding.values
            
            logger.warning("Gemini API returned an empty embedding vector on attempt %d", attempt)

        except APIError as api_err:
            logger.warning(
                "Gemini API error during embedding generation (attempt %d/%d): %s",
                attempt,
                max_retries,
                api_err,
            )
            if attempt == max_retries:
                raise api_err
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)

        except Exception as exc:
            logger.error("Unexpected error generating embedding: %s", exc, exc_info=True)
            raise exc

    return []


async def get_batch_embeddings(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 10,
) -> List[List[float]]:
    """
    Processes a list of text strings in parallel batches to optimize embedding generation speed.
    """
    if not texts:
        return []

    logger.info("Generating embeddings for %d items in batches of %d...", len(texts), batch_size)
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tasks = [get_gemini_embedding(text, task_type=task_type) for text in batch]
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error("Failed to generate embedding for item %d in batch: %s", i + idx, result)
                all_embeddings.append([])
            else:
                all_embeddings.append(result)

    logger.info("Completed batch embedding generation for %d items.", len(texts))
    return all_embeddings


async def get_query_embedding(query: str) -> List[float]:
    """
    Helper function to embed user search queries. Uses RETRIEVAL_QUERY task_type
    for optimal cosine similarity matching against RETRIEVAL_DOCUMENT vectors.
    """
    return await get_gemini_embedding(
        text=query,
        task_type="RETRIEVAL_QUERY",
    )