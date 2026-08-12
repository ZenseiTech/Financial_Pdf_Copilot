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
    db: AsyncSession,
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    filename_filter: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Retrieves context from pgvector, constructs a grounded Gemini system prompt,
    and streams the model response tokens asynchronously.

    :param db: AsyncSession database connection.
    :param user_query: User's natural language prompt.
    :param conversation_history: Optional list of past messages [{'role': 'user'|'model', 'text': '...'}]
    :param filename_filter: Optional document filter scope.
    :yields: Streaming text chunks from Gemini.
    """
    logger.info("Processing query for streaming agent: '%s'", user_query)

    # 1. Retrieve RAG Context Chunks from pgvector
    retrieved_chunks = await search_similar_chunks(
        db=db,
        query=user_query,
        filename_filter=filename_filter,
    )

    # 2. Format Context Block
    context_text = format_context_for_prompt(retrieved_chunks)
    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(context_block=context_text)

    # 3. Construct Contents Payload (History + Current Query)
    contents: List[types.Content] = []

    if conversation_history:
        for msg in conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["text"])],
                )
            )

    # Append current user prompt
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
    )

    # 4. Configure Gemini Generation Parameters
    gen_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=config.gemini.temperature,
        max_output_tokens=config.gemini.max_output_tokens,
    )

    logger.debug("Initiating streaming request with Gemini model '%s'...", config.gemini.model)

    try:
        # 5. Call Async Streaming Endpoint via Google GenAI SDK
        response_stream = await client.aio.models.generate_content_stream(
            model=config.gemini.model,
            contents=contents,
            config=gen_config,
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    except APIError as api_err:
        logger.error("Gemini API stream error: %s", api_err, exc_info=True)
        yield f"\n[API Error: Unable to complete request due to Gemini service error: {api_err.message}]"
    except Exception as exc:
        logger.error("Unexpected error in agent stream loop: %s", exc, exc_info=True)
        yield "\n[System Error: Internal processing error occurred while generating response.]"


async def generate_agent_response(
    db: AsyncSession,
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    filename_filter: Optional[str] = None,
) -> str:
    """
    Non-streaming variant that accumulates and returns the full response string.
    """
    full_response = []
    async for chunk in stream_agent_response(
        db=db,
        user_query=user_query,
        conversation_history=conversation_history,
        filename_filter=filename_filter,
    ):
        full_response.append(chunk)

    return "".join(full_response)