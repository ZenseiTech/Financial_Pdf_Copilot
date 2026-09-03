import logging.config
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


class AppSettings(BaseModel):
    name: str = "Internal Developer Copilot"
    version: str = "1.0.0"
    env: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


class DatabaseSettings(BaseModel):
    url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5431/dev_docs_db"
    )


class GeminiSettings(BaseModel):
    model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-004"
    embedding_dimension: int = 768
    temperature: float = 0.2
    max_output_tokens: int = 2048


class RAGSettings(BaseModel):
    top_k: int = 5
    similarity_threshold: float = 0.65


class Settings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    logging: Dict[str, Any] = Field(default_factory=dict)


def load_settings() -> Settings:
    """Loads settings from config.yaml with fallback to defaults."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    # Map 'database' section from YAML to 'db' attribute on Settings
    if "database" in raw_dict and "db" not in raw_dict:
        raw_dict["db"] = raw_dict.pop("database")

    return Settings(**raw_dict)


# Global settings instance
config = load_settings()


def init_logging() -> None:
    """Initializes logging configuration from the logging dict in settings."""
    if config.logging:
        logging.config.dictConfig(config.logging)
    else:
        logging.basicConfig(level=logging.INFO)
