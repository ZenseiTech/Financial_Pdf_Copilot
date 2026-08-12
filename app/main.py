from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_v1_router
from app.core.config import config, init_logging

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize logging dictConfig
    init_logging()
    logger.info("Starting up %s (v%s)...", config.app.name, config.app.version)

    yield

    logger.info("Shutting down %s...", config.app.name)


# 2. Instantiate FastAPI Application
app = FastAPI(
    title=config.app.name,
    version=config.app.version,
    debug=config.app.debug,
    lifespan=lifespan,
)

# 3. Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Include Versioned API Routes
app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "app": config.app.name, "version": config.app.version}