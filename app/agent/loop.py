import logging
from typing import AsyncGenerator, Dict, List, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.rag.retriever import format_context_for_prompt, search_similar_chunks

logger = logging.getLogger(__name__)

# Initialize Google GenAI client (uses GEMINI_API_KEY environment variable)
client = genai.Client()

SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert Financial Documentation Assistant. Your goal is to provide accurate, concise, and professional answers based on provided financial context (10-Ks, 10-Qs, income statements, balance sheets).

### Guidelines:
1. Base your answers strictly on the provided Context Chunks below.
2. If the answer cannot be determined from the context, state that clearly rather than hallucinating financial figures.
3. When referencing financial figures, specify the exact units, years, and table sources where applicable.
4. Cite the source document and page number for key findings.

### Retained Context:
{context_block}
"""


async def stream_agent_response(
    db: Optional[AsyncSession] = None,
    user_query: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    filename_filter: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    context_str = ""

    # 1. Perform RAG vector search if DB session exists
    if db is not None:
        try:
            retrieved_chunks = await search_similar_chunks(
                session=db,
                query=user_query,
                top_k=3,
            )
            if retrieved_chunks:
                context_str = "\n\n".join(
                    [
                        f"[Source: {c.get('filename', 'Doc')}]\n{c['content']}"
                        for c in retrieved_chunks
                    ]
                )
        except Exception as err:
            logger.warning("Retrieval skipped due to error: %s", err)

    # 2. Construct final prompt
    if context_str:
        final_prompt = f"Context:\n{context_str}\n\nUser Question:\n{user_query}"
    else:
        final_prompt = user_query

    # 3. Create an AsyncChat session to handle AFC and streaming cleanly
    chat = client.aio.chats.create(model=config.gemini.model)

    # 4. Stream message response via AsyncChat
    response_stream = await chat.send_message_stream(final_prompt)

    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text


def generate_agent_response(
    db: Optional[AsyncSession] = None,
    user_query: Optional[str] = None,
    prompt: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    filename_filter: Optional[str] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    RAG agent entrypoint supporting keyword or positional prompt calls.
    """
    # If the first positional argument is a string instead of an AsyncSession, treat it as query text
    query_text = None
    if isinstance(db, str):
        query_text = db
        db = None
    else:
        query_text = prompt or user_query or kwargs.get("query")

    if not query_text:
        raise ValueError("Either 'user_query' or 'prompt' must be provided.")

    return stream_agent_response(
        db=db,
        user_query=query_text,
        conversation_history=conversation_history,
        filename_filter=filename_filter,
    )
