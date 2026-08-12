# app/db/__init__.py
from app.db.base import Base
from app.rag.indexer import DocumentChunkModel  # Import all ORM models here

__all__ = ["Base", "DocumentChunkModel"]