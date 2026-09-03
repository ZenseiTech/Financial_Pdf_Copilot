# app/db/__init__.py
from app.db.base import Base
from app.db.models import DocumentChunkModel  # Import all ORM models here

__all__ = ["Base", "DocumentChunkModel"]
