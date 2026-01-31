"""
IMSKOS Backend - Pydantic Models (Schemas)
Request and response models for the API.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class QuerySource(str, Enum):
    """Available query sources for routing."""
    AUTO = "auto"
    VECTOR = "vector"
    WIKIPEDIA = "wikipedia"
    WEB = "web"


class JobStatus(str, Enum):
    """Job status for async operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Query Models
# ============================================================================

class QueryOptions(BaseModel):
    """Advanced query options."""
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")
    hyde: bool = Field(default=False, description="Enable HyDE (Hypothetical Document Embeddings)")


class QueryRequest(BaseModel):
    """Request model for POST /api/v1/query."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query text")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    source: QuerySource = Field(default=QuerySource.AUTO, description="Query source routing")
    options: QueryOptions = Field(default_factory=QueryOptions, description="Advanced options")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the types of agent memory?",
                    "user_id": "user-123",
                    "source": "auto",
                    "options": {"top_k": 5, "temperature": 0.0, "hyde": False}
                }
            ]
        }
    }


class SourceResult(BaseModel):
    """Individual source result in query response."""
    source_id: str = Field(..., description="Unique source document ID")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    snippet: str = Field(..., description="Relevant text snippet")
    url: Optional[str] = Field(default=None, description="Source document URL")


class QueryMetrics(BaseModel):
    """Performance metrics for query execution."""
    elapsed_ms: int = Field(..., description="Total execution time in milliseconds")
    tokens: int = Field(..., description="Number of tokens in response")
    embedding_ms: Optional[int] = Field(default=None, description="Embedding generation time")
    retrieval_ms: Optional[int] = Field(default=None, description="Vector retrieval time")
    llm_ms: Optional[int] = Field(default=None, description="LLM inference time")


class QueryResponse(BaseModel):
    """Response model for POST /api/v1/query."""
    id: str = Field(..., description="Unique query ID")
    response: str = Field(..., description="Generated response text")
    sources: list[SourceResult] = Field(default_factory=list, description="Source documents")
    metrics: QueryMetrics = Field(..., description="Performance metrics")
    routing_reason: str = Field(..., description="Explanation for routing decision")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "mock-query-0001",
                    "response": "This is a deterministic mock answer from IMSKOS scaffold.",
                    "sources": [
                        {
                            "source_id": "doc-1",
                            "similarity_score": 0.92,
                            "snippet": "Example matched text...",
                            "url": "/storage/docs/doc1.pdf"
                        }
                    ],
                    "metrics": {"elapsed_ms": 75, "tokens": 34},
                    "routing_reason": "mock: scaffold route to vector (demo)"
                }
            ]
        }
    }


# ============================================================================
# Upload/Indexing Models
# ============================================================================

class UploadResponse(BaseModel):
    """Response model for POST /api/v1/upload."""
    file_id: str = Field(..., description="Unique file identifier")
    job_id: str = Field(..., description="Indexing job identifier")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IndexRequest(BaseModel):
    """Request model for POST /api/v1/index."""
    file_id: str = Field(..., description="File ID to index")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Chunk overlap in tokens")


class IndexJobResponse(BaseModel):
    """Response model for indexing job status."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    logs: list[str] = Field(default_factory=list, description="Processing logs")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# ============================================================================
# Document Models
# ============================================================================

class DocumentMetadata(BaseModel):
    """Document metadata."""
    doc_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File MIME type")
    size_bytes: int = Field(..., description="File size")
    chunk_count: int = Field(default=0, description="Number of chunks")
    indexed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    url: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for GET /api/v1/docs."""
    documents: list[DocumentMetadata] = Field(default_factory=list)
    total: int = Field(default=0)


# ============================================================================
# Feedback Models
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request model for POST /api/v1/feedback."""
    query_id: str = Field(..., description="Associated query ID")
    result_id: str = Field(..., description="Result ID being rated")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    score: int = Field(..., ge=-1, le=1, description="Feedback score: -1 (down), 0 (neutral), 1 (up)")
    note: Optional[str] = Field(default=None, max_length=1000, description="Optional feedback note")


# ============================================================================
# Health Check Models
# ============================================================================

class HealthResponse(BaseModel):
    """Response model for GET /health."""
    status: str = Field(default="healthy")
    version: str
    environment: str
    mock_mode: dict[str, bool] = Field(default_factory=dict, description="Services in mock mode")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
