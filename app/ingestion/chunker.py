import logging
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
)


def process_pdf_to_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text and tables locally from a PDF.
    Returns a list of dicts: [{'content': str, 'metadata': dict}, ...]
    """
    chunks: List[Dict[str, Any]] = []
    file_name = Path(pdf_path).name

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1

            # Extract tables into Markdown
            tables = page.extract_tables()
            for table in tables:
                if not table or not any(table):
                    continue
                try:
                    df = pd.DataFrame(table[1:], columns=table[0]).dropna(how="all")
                    markdown_table = df.to_markdown(index=False)
                    if markdown_table and markdown_table.strip():
                        chunks.append(
                            {
                                "content": markdown_table,
                                "metadata": {
                                    "file_name": file_name,
                                    "page_number": page_num,
                                    "chunk_type": "table",
                                },
                            }
                        )
                except Exception as err:
                    logger.warning("Table parse error on page %d: %s", page_num, err)

            # Extract prose text
            text = page.extract_text()
            if text and text.strip():
                sub_chunks = text_splitter.split_text(text)
                for sub_chunk in sub_chunks:
                    chunks.append(
                        {
                            "content": sub_chunk,
                            "metadata": {
                                "file_name": file_name,
                                "page_number": page_num,
                                "chunk_type": "text",
                            },
                        }
                    )

    return chunks
