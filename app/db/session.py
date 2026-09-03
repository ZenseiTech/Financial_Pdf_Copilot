# app/db/session.py
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import config

logger = logging.getLogger(__name__)

# Create the async engine
engine = create_async_engine(
    config.database.url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db():
    """FastAPI dependency for yielding database sessions."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """
    Initializes database extensions and creates missing tables dynamically.
    Imports models locally inside the function to eliminate circular imports.
    """
    # Local import inside the function breaks the circular dependency chain
    from app.db.models import Base

    async with engine.begin() as conn:
        # Enable vector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Create all tables registered in Base metadata
        await conn.run_sync(Base.metadata.create_all)

    logger.info("✓ Database initialized and pgvector extension verified.")
