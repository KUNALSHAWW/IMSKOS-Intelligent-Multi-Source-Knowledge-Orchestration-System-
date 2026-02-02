"""
IMSKOS Backend - Index API Router (v1)
Handles document indexing with async background processing.

Key endpoints:
- POST /api/v1/index - Start indexing task
- GET /api/v1/index/status/{task_id} - Poll task status
- POST /api/v1/index/upload - Upload and index files
"""
import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.schemas import (
    IndexRequest,
    IndexResponse,
    IndexUploadResponse,
    TaskStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Index"])

# Storage path for uploaded files
STORAGE_PATH = os.getenv("STORAGE_PATH", "/app/storage")

# Maximum upload file size (10 MB)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))

# Check if we should skip Celery entirely (no Redis)
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"


def get_celery_app():
    """Get Celery app, handling import errors gracefully."""
    # If in mock mode, don't even try to use Celery
    if MOCK_MODE:
        logger.info("MOCK_MODE enabled, skipping Celery")
        return None, None
    
    try:
        from app.worker import celery_app
        from app.worker.tasks import index_documents_task
        return celery_app, index_documents_task
    except ImportError as e:
        logger.warning(f"Celery not available: {e}")
        return None, None


def _create_mock_response(document_ids: List[str], reason: str = "Celery unavailable") -> IndexResponse:
    """Create a mock response for when Celery is not available."""
    mock_task_id = f"mock-{uuid.uuid4().hex[:12]}"
    return IndexResponse(
        task_id=mock_task_id,
        status="pending",
        message=f"[MOCK] Indexing queued for {len(document_ids)} documents ({reason})",
        created_at=datetime.utcnow()
    )


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Start document indexing",
    description="""
    Start document indexing as a background task.
    
    Returns immediately with a task_id that can be used to poll for status.
    This prevents the UI from blocking on long-running indexing operations.
    
    **Workflow:**
    1. Call POST /api/v1/index with document_ids and/or URLs
    2. Receive task_id immediately
    3. Poll GET /api/v1/index/status/{task_id} for progress
    4. Task completes with success or error
    """
)
async def start_indexing(request: IndexRequest) -> IndexResponse:
    """
    Start document indexing as a background task.
    
    The actual indexing runs in a Celery worker, allowing this endpoint
    to return immediately without blocking.
    """
    settings = get_settings()
    
    # Log mock mode status
    for warning in settings.log_mock_mode_warnings():
        logger.warning(warning)
    
    # Get Celery components
    celery_app, index_documents_task = get_celery_app()
    
    if celery_app is None or index_documents_task is None:
        # Fallback: Return mock response when Celery is not available
        logger.warning("Celery not available, returning mock response")
        return _create_mock_response(request.document_ids, "Celery unavailable")
    
    try:
        # Submit task to Celery worker
        task = index_documents_task.delay(
            document_ids=request.document_ids,
            urls=request.urls,
            options=request.options.model_dump() if request.options else None
        )
        
        logger.info(f"Started indexing task {task.id} for {len(request.document_ids)} documents")
        
        return IndexResponse(
            task_id=task.id,
            status="pending",
            message=f"Indexing started for {len(request.document_ids)} documents",
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        # If Celery/Redis fails, fall back to mock mode instead of crashing
        error_msg = str(e).lower()
        if "redis" in error_msg or "connection" in error_msg or "retry" in error_msg:
            logger.warning(f"Celery/Redis unavailable, falling back to mock mode: {e}")
            return _create_mock_response(request.document_ids, "Redis/Worker unavailable - running in mock mode")
        
        logger.exception("Failed to start indexing task")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start indexing: {str(e)}"
        )


@router.get(
    "/index/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get indexing task status",
    description="""
    Get the status of an indexing task.
    
    Poll this endpoint to track progress of long-running indexing operations.
    
    **Status values:**
    - `PENDING`: Task is queued but not started
    - `STARTED`: Task has been picked up by a worker
    - `PROGRESS`: Task is running, check `progress` for details
    - `SUCCESS`: Task completed successfully
    - `FAILURE`: Task failed, check `error` for details
    """
)
async def get_index_status(task_id: str) -> TaskStatusResponse:
    """
    Get the status of an indexing task.
    """
    # Handle mock task IDs
    if task_id.startswith("mock-"):
        return TaskStatusResponse(
            task_id=task_id,
            status="SUCCESS",
            progress={"current": 1, "total": 1, "status": "[MOCK] Complete"},
            result={
                "processed": 1,
                "failed": 0,
                "chunks_created": 10,
                "mock_mode": True
            }
        )
    
    celery_app, index_documents_task = get_celery_app()
    
    if celery_app is None or index_documents_task is None:
        return TaskStatusResponse(
            task_id=task_id,
            status="UNKNOWN",
            error="Celery not available"
        )
    
    try:
        result = index_documents_task.AsyncResult(task_id)
        
        response = TaskStatusResponse(
            task_id=task_id,
            status=result.status
        )
        
        if result.status == "PROGRESS":
            response.progress = result.info
        elif result.status == "SUCCESS":
            response.result = result.result
        elif result.status == "FAILURE":
            response.error = str(result.result) if result.result else "Unknown error"
        elif result.status == "STARTED":
            response.progress = {"status": "Task started, processing..."}
            
        return response
        
    except Exception as e:
        logger.exception(f"Failed to get status for task {task_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.post(
    "/index/upload",
    response_model=IndexUploadResponse,
    summary="Upload and index files",
    description="""
    Upload files and queue them for indexing.
    
    Files are saved to storage and a background task is created for indexing.
    Returns immediately with task_id for status polling.
    
    **Supported formats:** PDF, TXT, MD, DOCX
    """
)
async def upload_and_index(
    files: List[UploadFile] = File(..., description="Files to upload and index")
) -> IndexUploadResponse:
    """
    Upload files and queue them for indexing.
    """
    settings = get_settings()
    
    # Ensure storage directory exists
    os.makedirs(STORAGE_PATH, exist_ok=True)
    
    document_ids = []
    uploaded_files = []
    
    try:
        for file in files:
            # Generate unique document ID
            doc_id = f"doc-{uuid.uuid4().hex[:12]}"
            
            # Sanitize filename - ensure we have a valid name
            safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
            if not safe_filename:
                # Fallback: use doc_id with original extension
                _, ext = os.path.splitext(file.filename) if file.filename else ("", ".bin")
                safe_filename = f"{doc_id}{ext or '.bin'}"
            
            # Read content in chunks to enforce size limit
            content = b""
            chunk_size = 64 * 1024  # 64KB chunks
            total_read = 0
            
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
                    )
                content += chunk
            
            # Save file asynchronously
            file_path = os.path.join(STORAGE_PATH, f"{doc_id}_{safe_filename}")
            
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)
            
            document_ids.append(doc_id)
            # Only return safe metadata, not internal file_path
            uploaded_files.append({
                "doc_id": doc_id,
                "filename": file.filename,
                "size": len(content)
            })
            
            logger.info(f"Saved uploaded file {file.filename} as {doc_id}")
        
        # Queue indexing task
        celery_app, index_documents_task = get_celery_app()
        
        if celery_app is None or index_documents_task is None:
            # Mock mode
            mock_task_id = f"mock-upload-{uuid.uuid4().hex[:8]}"
            return IndexUploadResponse(
                message=f"[MOCK] Uploaded {len(files)} files",
                document_ids=document_ids,
                task_id=mock_task_id,
                status="pending",
                files=uploaded_files
            )
        
        task = index_documents_task.delay(document_ids=document_ids)
        
        return IndexUploadResponse(
            message=f"Uploaded {len(files)} files and started indexing",
            document_ids=document_ids,
            task_id=task.id,
            status="pending",
            files=uploaded_files
        )
        
    except Exception as e:
        logger.exception("Failed to upload and index files")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


@router.get(
    "/index/health",
    summary="Check indexing service health",
    description="Check if the indexing worker is available and healthy."
)
async def index_health():
    """
    Check the health of the indexing service.
    """
    celery_app, _ = get_celery_app()
    
    health_status = {
        "celery_available": celery_app is not None,
        "redis_connected": False,
        "worker_count": 0,
        "workers": []
    }
    
    if celery_app is not None:
        try:
            # Check Redis connection
            inspect = celery_app.control.inspect()
            active_workers = inspect.active()
            
            # inspect.active() returns None on timeout/no response
            if active_workers is None:
                logger.warning("Celery inspect.active() returned None - workers may be unavailable")
                health_status["redis_connected"] = False
                health_status["worker_count"] = 0
                health_status["workers"] = []
            else:
                # active_workers is a dict when workers are available
                health_status["redis_connected"] = True
                health_status["worker_count"] = len(active_workers)
                health_status["workers"] = list(active_workers.keys())
                
        except Exception as e:
            logger.warning(f"Failed to inspect Celery workers: {e}")
            health_status["error"] = str(e)
    
    return health_status
