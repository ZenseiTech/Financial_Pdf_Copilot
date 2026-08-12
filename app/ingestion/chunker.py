import logging
import re
from typing import Any, Dict, List

from app.core.config import config

logger = logging.getLogger(__name__)


def chunk_text_section(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Splits narrative text into overlapping word chunks based on sentence boundaries
    where possible, avoiding mid-sentence cuts.
    """
    if not text or not text.strip():
        return []

    # Split text by sentence terminators
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if current_word_count + word_count > chunk_size and current_chunk:
            # Join current sentence collection into a chunk
            chunk_str = " ".join(current_chunk)
            chunks.append(chunk_str)

            # Apply overlap by retaining the tail words of the previous chunk
            overlap_words = chunk_str.split()[-chunk_overlap:]
            current_chunk = overlap_words + words
            current_word_count = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def format_table_as_markdown(table: Dict[str, Any]) -> str:
    """
    Converts raw extracted table dicts (headers + rows) into a structured Markdown string
    so LLM vector embeddings retain column and row context cleanly.
    """
    headers: List[str] = table.get("headers", [])
    rows: List[List[str]] = table.get("rows", [])
    title: str = table.get("title", "Financial Data Table")

    if not rows:
        return ""

    markdown_lines = [f"### Table: {title}"]

    # Render Headers
    if headers:
        markdown_lines.append("| " + " | ".join(headers) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    else:
        # Generate default headers if missing
        col_count = max(len(r) for r in rows)
        markdown_lines.append("| " + " | ".join([f"Col {i+1}" for i in range(col_count)]) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * col_count) + " |")

    # Render Rows
    for row in rows:
        formatted_row = [str(cell).strip().replace("\n", " ") if cell else "-" for cell in row]
        markdown_lines.append("| " + " | ".join(formatted_row) + " |")

    return "\n".join(markdown_lines)


def chunk_financial_document(
    text: str,
    tables: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Primary chunking pipeline for financial reports.

    1. Processes narrative text into overlapping chunks.
    2. Treats each financial table as an atomic, indivisible Markdown chunk.
    3. Attaches contextual metadata to every chunk for vector retrieval.
    """
    chunks: List[Dict[str, Any]] = []

    # 1. Process narrative text chunks
    text_chunks = chunk_text_section(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for idx, chunk_str in enumerate(text_chunks):
        chunks.append({
            "content": chunk_str,
            "chunk_type": "text",
            "chunk_index": idx,
            "metadata": {
                **metadata,
                "type": "text",
                "chunk_id": f"{metadata.get('filename', 'doc')}_text_{idx}",
            },
        })

    # 2. Process financial tables as atomic Markdown units
    for idx, table in enumerate(tables):
        table_md = format_table_as_markdown(table)
        if not table_md.strip():
            continue

        table_metadata = {
            **metadata,
            "type": "table",
            "page_number": table.get("page", 1),
            "table_title": table.get("title", f"Table {idx + 1}"),
            "chunk_id": f"{metadata.get('filename', 'doc')}_table_{idx}",
        }

        chunks.append({
            "content": table_md,
            "chunk_type": "table",
            "chunk_index": len(text_chunks) + idx,
            "metadata": table_metadata,
        })

    logger.info(
        "Chunked document '%s': %d text chunks, %d table chunks",
        metadata.get("filename", "unknown"),
        len(text_chunks),
        len(tables),
    )

    return chunks