import asyncio
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import config

logger = logging.getLogger(__name__)

_model_instance = None


def get_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        # Uses local HuggingFace model (BAAI/bge-base-en-v1.5 = 768 dims)
        _model_instance = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _model_instance


def _encode_sync(texts: List[str], is_query: bool = False) -> List[List[float]]:
    model = get_model()
    if is_query:
        texts = [
            f"Represent this sentence for searching relevant passages: {t}"
            for t in texts
        ]
    return model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


async def get_gemini_embedding(
    text: str, task_type: str = "RETRIEVAL_DOCUMENT", **kwargs
) -> List[float]:
    if not text or not text.strip():
        return []
    is_query = task_type == "RETRIEVAL_QUERY"
    results = await asyncio.to_thread(_encode_sync, [text], is_query)
    return results[0] if results else []


async def get_batch_embeddings(
    texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT", **kwargs
) -> List[List[float]]:
    if not texts:
        return []
    is_query = task_type == "RETRIEVAL_QUERY"
    return await asyncio.to_thread(_encode_sync, texts, is_query)


async def get_query_embedding(query: str) -> List[float]:
    return await get_gemini_embedding(text=query, task_type="RETRIEVAL_QUERY")
