# Internal Developer Copilot

A high-performance Retrieval-Augmented Generation (RAG) backend service engineered for internal financial document analysis (e.g., 10-K, 10-Q reports). The system processes narrative text and structured financial tables, generates vector embeddings using the Google GenAI SDK, stores context in PostgreSQL via pgvector, and streams answers to clients over Server-Sent Events (SSE).

## Key Features

    * Async FastAPI Engine: Fully asynchronous HTTP endpoints utilizing asyncpg and SQLAlchemy 2.0.

    * Table-Aware Ingestion Pipeline: Uses pdfplumber to extract narrative text while preserving financial tables as unbroken Markdown blocks.

    * Gemini Embeddings & Generation: Integrates google-genai SDK for vector embeddings (text-embedding-004) and streaming chat completions (gemini-2.5-flash).

    * Vector Search with pgvector: Stores and queries embeddings in PostgreSQL using cosine similarity (1 - cosine_distance) with optional section or filename metadata scoping.

    * Real-time SSE Streaming: Delivers token chunks over Server-Sent Events with client disconnect detection and Nginx proxy support.


    ┌────────────────────────────────────────────────────────────────────────┐
    │                      FINANCIAL PDF INGESTION PIPELINE                  │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ pdfplumber / Table Extraction
    ┌───────────────────────────────────▼────────────────────────────────────┐
    │  Postgres + pgvector Database                                          │
    │  • Text Chunks (text-embedding-004) + Table JSON Metadata              │
    │  • Metadata: { ticker: "AAPL", period: "Q3-2026", page_number: 42 }   │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Query
    ┌───────────────────────────────────▼────────────────────────────────────┐
    │  1. GUARDRAILS & CACHE (app/guardrails/financial_filter.py & Redis)   │
    │     • Intercept PII & unverified ticker symbols                        │
    │     • Sub-10ms Redis Semantic Query Cache                              │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Cache Miss
    ┌───────────────────────────────────▼────────────────────────────────────┐
    │  2. GEMINI 2.5 FLASH AGENT + NATIVE CODE EXECUTION                     │
    │     • Hybrid pgvector Retrieval (RRF Dense + Sparse BM25)               │
    │     • Executes Python code sandbox for deterministic math (CAGR, %s)   │
    │     • Returns Pydantic Structured Output with Source Citations         │
    └────────────────────────────────────────────────────────────────────────┘

## Project Directory Structure

    .
    ├── app/
    │   ├── agent/
    │   │   └── loop.py          # Gemini streaming agent loop & prompt context injection
    │   ├── api/
    │   │   └── v1/
    │   │       ├── chat.py      # SSE streaming endpoint (/api/v1/chat/stream)
    │   │       └── router.py    # Primary API v1 router
    │   ├── core/
    │   │   └── config.py        # Settings loader (YAML + Pydantic v2)
    │   ├── db/
    │   │   ├── base.py          # SQLAlchemy 2.0 DeclarativeBase
    │   │   └── session.py       # Async engine & session dependency
    │   ├── ingestion/
    │   │   ├── chunker.py       # Sentence splitter & Markdown table preservation
    │   │   ├── main.py          # Ingestion CLI / runner entrypoint
    │   │   └── pdf_parser.py    # Async pdfplumber text & table extraction
    │   ├── rag/
    │   │   ├── embeddings.py    # Google GenAI SDK embedding client wrapper
    │   │   ├── indexer.py       # DocumentChunkModel schema & database indexing
    │   │   └── retriever.py     # Cosine similarity search against pgvector
    │   └── main.py              # Root FastAPI web application
    ├── config.yaml              # Global application configuration
    ├── requirements.txt         # Project dependencies
    └── test_pipeline.py         # End-to-end verification script

## Prerequisites

    Python: 3.10+

    Database: PostgreSQL with the vector extension enabled (pgvector)

    API Key: A valid Google Gemini API Key

## Getting Started

### 1. Clone & Setup Virtual Environment

        git clone https://github.com/your-org/internal-dev-copilot.git
        cd internal-dev-copilot

        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt

### 2. Configure Environment & Database

Update config.yaml or set environment variables for database credentials:

YAML
-> config.yaml

    database:
    url: "postgresql+asyncpg://postgres:postgres@localhost:5432/dev_docs_db"

Set your Gemini API Key in your shell environment:

Bash
export GEMINI_API_KEY="your-gemini-api-key"
Ensure pgvector is available on your PostgreSQL instance:

SQL
CREATE EXTENSION IF NOT EXISTS vector;

### How to Run

Run as a module from project root:

    python -m app.ingestion.main --file data/q3_financials.pdf

Batch process a directory:

    python -m app.ingestion.main --dir data/annual_reports/

### How to Run the Verification Script

Make sure your PostgreSQL database with pgvector extension is running.

Set your Google Gemini API key:

    export GEMINI_API_KEY="your-gemini-api-key"

Run the test script:

    python test_pipeline.py
