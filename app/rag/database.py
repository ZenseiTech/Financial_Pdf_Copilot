from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text, String, Integer, Index, text
from pgvector.sqlalchemy import Vector

DATABASE_URL = "postgresql+asyncpg://ai_user:ai_password@localhost:5432/financial_db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class FinancialChunk(Base):
    __tablename__ = "financial_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)      # e.g., "AAPL"
    filing_type: Mapped[str] = mapped_column(String(20), index=True) # e.g., "10-K"
    fiscal_year: Mapped[int] = mapped_column(Integer, index=True)    # e.g., 2026
    page_number: Mapped[int] = mapped_column(Integer)
    is_table: Mapped[bool] = mapped_column(default=False)
    content: Mapped[str] = mapped_column(Text)
    
    # 768 dimensions matching text-embedding-004
    embedding: Mapped[list[float]] = mapped_column(Vector(768))

async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
        await conn.run_sync(Base.metadata.create_all)
        
        # HNSW Index for fast vector similarity search
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_financial_hnsw 
            ON financial_chunks USING hnsw (embedding vector_cosine_ops);
        """))