"""
IMSKOS Backend - Query Router (v1)
Handles POST /api/v1/query with deterministic mock responses for scaffold.
"""
import logging
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.schemas import (
    QueryMetrics,
    QueryRequest,
    QueryResponse,
    SourceResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Query"])

# Deterministic mock response for scaffold
MOCK_RESPONSE = QueryResponse(
    id="mock-query-0001",
    response="This is a deterministic mock answer from IMSKOS scaffold.",
    sources=[
        SourceResult(
            source_id="doc-1",
            similarity_score=0.92,
            snippet="Example matched text...",
            url="/storage/docs/doc1.pdf"
        )
    ],
    metrics=QueryMetrics(
        elapsed_ms=75,
        tokens=34,
        embedding_ms=15,
        retrieval_ms=35,
        llm_ms=25
    ),
    routing_reason="mock: scaffold route to vector (demo)"
)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Execute intelligent query",
    description="""
    Execute an intelligent query against the IMSKOS knowledge base.
    
    The query router automatically determines the optimal data source:
    - **auto**: LLM-powered routing to best source
    - **vector**: Force vector store search (Astra DB)
    - **wikipedia**: Force Wikipedia search
    - **web**: Force web search
    
    **Note**: In scaffold mode, returns deterministic mock response.
    """
)
async def execute_query(request: QueryRequest) -> QueryResponse:
    """
    Execute a query and return results with sources and metrics.
    
    In scaffold mode, returns a deterministic mock response.
    Full implementation will include:
    - Groq LLM routing (with fallback heuristics)
    - Vector store retrieval (Astra DB)
    - Wikipedia/web search integration
    - SSE streaming support
    """
    settings = get_settings()
    start_time = time.time()
    
    # Log mock mode warnings
    for warning in settings.log_mock_mode_warnings():
        logger.warning(warning)
    
    # Generate unique query ID
    query_id = f"query-{uuid.uuid4().hex[:8]}"
    
    # Calculate elapsed time
    elapsed_ms = int((time.time() - start_time) * 1000) + 75  # Add mock processing time
    
    # Return mock response with updated ID and metrics
    return QueryResponse(
        id=query_id,
        response=MOCK_RESPONSE.response,
        sources=MOCK_RESPONSE.sources,
        metrics=QueryMetrics(
            elapsed_ms=elapsed_ms,
            tokens=MOCK_RESPONSE.metrics.tokens,
            embedding_ms=MOCK_RESPONSE.metrics.embedding_ms,
            retrieval_ms=MOCK_RESPONSE.metrics.retrieval_ms,
            llm_ms=MOCK_RESPONSE.metrics.llm_ms
        ),
        routing_reason=f"mock: scaffold route to {request.source.value} (demo)"
    )
