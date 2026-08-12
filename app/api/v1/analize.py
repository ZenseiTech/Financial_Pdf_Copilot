import hashlib
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from google import genai
from google.genai import types

from app.rag.database import AsyncSessionLocal
from app.core.redis import redis_manager

router = APIRouter()
client = genai.Client()

# Pydantic Schema for Auditability and Structured Output
class Citation(BaseModel):
    page_number: int = Field(description="Page number in the PDF filing where the source data was found")
    source_snippet: str = Field(description="Exact line or snippet from the filing table/text")

class FinancialAnalysisResponse(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    executive_summary: str = Field(description="High-level narrative answer to the financial question")
    key_metrics: List[str] = Field(description="Extracted financial metrics or calculated values")
    calculated_code_output: Optional[str] = Field(None, description="Result of Python execution for math checks")
    citations: List[Citation] = Field(description="List of verified page-level citations")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.post("/analyze", response_model=FinancialAnalysisResponse)
async def analyze_financial_doc(
    ticker: str = Query(..., example="AAPL"),
    question: str = Query(..., example="What was the YoY growth rate in Net Income for 2026?"),
    db: AsyncSession = Depends(get_db)
):
    # 1. Check Redis Cache
    cache_key = hashlib.sha256(f"{ticker.upper()}:{question.strip().lower()}".encode()).hexdigest()
    cached_result = await redis_manager.get_cached_analysis(cache_key)
    if cached_result:
        return FinancialAnalysisResponse(**cached_result)

    # 2. Generate Query Embedding
    embed_response = await client.aio.models.embed_content(
        model="text-embedding-004",
        contents=question
    )
    query_vector = embed_response.embedding.values
    vector_str = f"[{','.join(map(str, query_vector))}]"

    # 3. Retrieve Context from Postgres using Ticker Metadata Filter
    sql = text("""
        SELECT page_number, content, is_table
        FROM financial_chunks
        WHERE ticker = :ticker
        ORDER BY embedding <=> :vector
        LIMIT 5;
    """)
    result = await db.execute(sql, {"ticker": ticker.upper(), "vector": vector_str})
    rows = result.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No financial filings found for ticker {ticker}")

    context_str = "\n\n".join([
        f"[Page {r.page_number} | Table={r.is_table}]\n{r.content}" for r in rows
    ])

    system_instruction = f"""You are a Chief Financial Officer (CFO) and Senior Equity Research Analyst.
Analyze the provided SEC filing context for {ticker.upper()} to answer the user's question accurately.

CRITICAL INSTRUCTIONS:
1. Always calculate math (YoY growth, margins, CAGR) using the Python Code Execution tool. Do NOT guess math.
2. Provide page-level citations for every statistic used.
"""

    # 4. Query Gemini 2.5 Flash with Code Execution & Structured Output Schema
    full_prompt = f"### Context:\n{context_str}\n\n### User Question:\n{question}"
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            # Enable Python execution sandbox for math
            tools=[types.Tool(code_execution=types.CodeExecution())],
            response_mime_type="application/json",
            response_schema=FinancialAnalysisResponse,
            temperature=0.0
        )
    )

    # 5. Parse Structured Output and Cache in Redis
    output_data = FinancialAnalysisResponse.model_validate_json(response.text)
    await redis_manager.set_cached_analysis(cache_key, output_data.model_dump())

    return output_data