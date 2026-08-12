from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from app.core.config import config

# Initialize the async engine with pooling parameters
async_engine = create_async_engine(
    config.db.url,  # e.g., "postgresql+asyncpg://user:pass@localhost:5432/dbname"
    echo=config.app.debug,
    pool_pre_ping=True,  # Test connections before returning them from the pool
    pool_size=10,
    max_overflow=20,
)

# Construct session factory bound to the async engine
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevents attribute expiration after commit in async context
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an AsyncSession and ensures proper cleanup."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()