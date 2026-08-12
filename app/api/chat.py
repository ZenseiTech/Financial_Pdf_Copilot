import json
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.agent.loop import stream_agent_response
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat Agent"])


class Message(BaseModel):
    role: str = Field(..., description="Message author role: 'user' or 'model'")
    text: str = Field(..., description="Text content of the message")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or prompt")
    history: Optional[List[Message]] = Field(
        default=[], description="Past conversation history"
    )
    filename_filter: Optional[str] = Field(
        default=None, description="Optional document filename filter for RAG retrieval"
    )


@router.post(
    "/stream",
    response_class=EventSourceResponse,
    summary="Stream Gemini Agent response using SSE",
    description="Executes RAG retrieval against pgvector and streams model response tokens back via Server-Sent Events.",
)
async def chat_stream_endpoint(
    request: Request,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    HTTP POST endpoint that returns a Server-Sent Events (SSE) stream.
    
    Event format:
    data: {"text": "chunk"}
    data: {"event": "end"}
    """
    # Convert Pydantic history objects to standard dict list
    conversation_history = [
        {"role": msg.role, "text": msg.text} for msg in payload.history
    ] if payload.history else None

    async def event_generator():
        try:
            # Yield initial acknowledgement signal
            yield {
                "event": "start",
                "data": json.dumps({"status": "processing", "query": payload.query}),
            }

            # Stream chunks from Gemini agent loop
            async for chunk_text in stream_agent_response(
                db=db,
                user_query=payload.query,
                conversation_history=conversation_history,
                filename_filter=payload.filename_filter,
            ):
                # Monitor for client disconnects during generation
                if await request.is_disconnected():
                    logger.warning("Client disconnected mid-stream for query: %s", payload.query)
                    break

                yield {
                    "event": "message",
                    "data": json.dumps({"text": chunk_text}),
                }

            # Yield completion event
            yield {
                "event": "end",
                "data": json.dumps({"status": "completed"}),
            }

        except Exception as exc:
            logger.error("Error during chat stream processing: %s", exc, exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"error": "An internal streaming error occurred."}),
            }

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables buffering on Nginx proxies
        },
    )