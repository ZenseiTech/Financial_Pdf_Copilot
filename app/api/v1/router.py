import logging
from fastapi import APIRouter

from app.api.v1.chat import router as chat_router

logger = logging.getLogger(__name__)

# Primary API Router for v1
api_v1_router = APIRouter()

# Register sub-routers
api_v1_router.include_router(chat_router)

# Example: Include health check or document ingestion endpoints here as needed
# from app.api.v1.health import router as health_router
# api_v1_router.include_router(health_router)