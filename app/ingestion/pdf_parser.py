import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber

logger = logging.getLogger(__name__)


def _extract_tables_from_page(page: pdfplumber.page.Page, page_number: int) -> List[Dict[str, Any]]:
    """
    Extracts structured tables from a single PDF page using pdfplumber's table extraction.
    Formats headers and rows cleanly.
    """
    extracted_tables: List[Dict[str, Any]] = []
    
    # Configure table extraction strategy suitable for financial reports
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "explicit_vertical_lines": [],
        "explicit_horizontal_lines": [],
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "min_words_vertical": 3,
        "min_words_horizontal": 1,
        "intersection_tolerance": 3,
    }

    # Extract tables (falls back to default strategy if explicit lines aren't found)
    raw_tables = page.extract_tables(table_settings=table_settings)
    if not raw_tables:
        raw_tables = page.extract_tables()

    for idx, table in enumerate(raw_tables):
        if not table or len(table) < 2:
            continue

        # Treat first non-empty row as headers
        headers = [str(cell).strip() if cell else "" for cell in table[0]]
        rows = []
        for row in table[1:]:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            # Skip completely empty rows
            if any(cleaned_row):
                rows.append(cleaned_row)

        if rows:
            extracted_tables.append({
                "page": page_number,
                "title": f"Page {page_number} Table {idx + 1}",
                "headers": headers,
                "rows": rows,
            })

    return extracted_tables


def _parse_pdf_sync(file_path: Path) -> Dict[str, Any]:
    """
    Synchronous PDF extraction execution using pdfplumber.
    Runs inside an async thread pool executor to prevent blocking the event loop.
    """
    full_text_pages: List[str] = []
    all_tables: List[Dict[str, Any]] = []

    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)
        logger.debug("Parsing PDF '%s' with %d pages...", file_path.name, page_count)

        for page_idx, page in enumerate(pdf.pages, start=1):
            # 1. Extract narrative text
            page_text = page.extract_text(layout=False) or ""
            if page_text.strip():
                full_text_pages.append(f"--- Page {page_idx} ---\n{page_text.strip()}")

            # 2. Extract structured tables
            tables = _extract_tables_from_page(page, page_number=page_idx)
            all_tables.extend(tables)

    combined_text = "\n\n".join(full_text_pages)

    return {
        "text": combined_text,
        "tables": all_tables,
        "page_count": page_count,
    }


async def extract_pdf_content(file_path: Path) -> Dict[str, Any]:
    """
    Async wrapper around pdfplumber parsing to ensure CPU-bound PDF reading
    does not block the FastAPI / asyncio event loop.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file does not exist at {path_obj}")

    logger.info("Parsing PDF file async: %s", path_obj.name)
    
    # Run the CPU-bound sync parsing in a separate thread
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _parse_pdf_sync, path_obj)

    logger.info(
        "Successfully extracted content from %s: %d pages, %d tables",
        path_obj.name,
        result["page_count"],
        len(result["tables"]),
    )

    return result