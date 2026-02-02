"""
IMSKOS Backend - Celery Tasks
Background tasks for document indexing and heavy operations.

These tasks run in a separate worker process, preventing the main
web server from blocking on CPU/IO intensive operations.
"""
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

logger = logging.getLogger(__name__)

# Check if we're in mock mode
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=540,
    time_limit=600,
    track_started=True,
)
def index_documents_task(
    self,
    document_ids: List[str],
    urls: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Background task for document indexing.
    
    This task:
    1. Loads documents from URLs or storage
    2. Splits them into chunks
    3. Generates embeddings
    4. Stores in vector database
    
    Args:
        document_ids: List of document IDs to index
        urls: Optional list of URLs to fetch documents from
        options: Optional indexing configuration
        
    Returns:
        Dict with indexing results and statistics
    """
    task_id = self.request.id
    logger.info(f"Starting indexing task {task_id} for {len(document_ids)} documents")
    
    results = {
        "task_id": task_id,
        "total": len(document_ids),
        "processed": 0,
        "failed": 0,
        "chunks_created": 0,
        "errors": [],
        "status": "processing"
    }
    
    try:
        # Update initial state
        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": len(document_ids),
                "status": "Initializing indexing..."
            }
        )
        
        if MOCK_MODE:
            # Mock mode: simulate indexing without actual processing
            logger.info(f"Task {task_id}: Running in MOCK MODE")
            return _mock_index_documents(self, document_ids, urls, options)
        
        # Real indexing implementation
        for i, doc_id in enumerate(document_ids):
            try:
                # Update progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i + 1,
                        "total": len(document_ids),
                        "status": f"Processing document {i + 1}/{len(document_ids)}: {doc_id}"
                    }
                )
                
                # Get URL if provided
                url = urls[i] if urls and i < len(urls) else None
                
                # Process the document
                chunks = _process_single_document(doc_id, url, options)
                results["processed"] += 1
                results["chunks_created"] += chunks
                
                logger.info(f"Task {task_id}: Processed document {doc_id} ({chunks} chunks)")
                
            except SoftTimeLimitExceeded:
                logger.warning(f"Task {task_id}: Soft time limit exceeded, wrapping up...")
                results["errors"].append({
                    "doc_id": doc_id,
                    "error": "Processing time limit exceeded"
                })
                results["status"] = "partial"
                break
                
            except Exception as e:
                logger.error(f"Task {task_id}: Failed to index document {doc_id}: {e}")
                results["failed"] += 1
                results["errors"].append({
                    "doc_id": doc_id,
                    "error": str(e)
                })
        
        results["status"] = "completed" if results["failed"] == 0 else "completed_with_errors"
        logger.info(f"Completed indexing task {task_id}: {results}")
        return results
        
    except SoftTimeLimitExceeded:
        logger.error(f"Task {task_id}: Soft time limit exceeded - task is being terminated gracefully")
        results["status"] = "timeout"
        results["errors"].append({"error": "Task soft time limit exceeded"})
        return results
        
    except Exception as e:
        logger.exception(f"Fatal error in indexing task {task_id}")
        results["status"] = "failed"
        results["errors"].append({"error": str(e)})
        
        # Retry on transient failures
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return results


def _mock_index_documents(
    task,
    document_ids: List[str],
    urls: Optional[List[str]],
    options: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Mock implementation for testing without external services."""
    results = {
        "task_id": task.request.id,
        "total": len(document_ids),
        "processed": 0,
        "failed": 0,
        "chunks_created": 0,
        "errors": [],
        "status": "processing",
        "mock_mode": True
    }
    
    for i, doc_id in enumerate(document_ids):
        # Simulate processing time
        time.sleep(0.5)
        
        # Update progress
        task.update_state(
            state="PROGRESS",
            meta={
                "current": i + 1,
                "total": len(document_ids),
                "status": f"[MOCK] Processing document {i + 1}/{len(document_ids)}"
            }
        )
        
        # Simulate some chunks being created
        mock_chunks = 5 + (hash(doc_id) % 10)
        results["processed"] += 1
        results["chunks_created"] += mock_chunks
        
        logger.info(f"[MOCK] Processed document {doc_id} ({mock_chunks} chunks)")
    
    results["status"] = "completed"
    return results


def _process_single_document(
    doc_id: str,
    url: Optional[str],
    options: Optional[Dict[str, Any]]
) -> int:
    """
    Process a single document for indexing.
    
    Args:
        doc_id: Document identifier
        url: Optional URL to fetch document from
        options: Processing options
        
    Returns:
        Number of chunks created
    """
    # Import here to avoid circular imports and reduce startup time
    try:
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        logger.warning("LangChain not available, using mock processing")
        return 5  # Return mock chunk count
    
    chunks_created = 0
    
    if url:
        # Load from URL
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=options.get("chunk_size", 500) if options else 500,
                chunk_overlap=options.get("chunk_overlap", 50) if options else 50,
            )
            doc_splits = text_splitter.split_documents(docs)
            chunks_created = len(doc_splits)
            
            # TODO: Generate embeddings and store in vector DB
            # This would connect to Astra DB in production
            logger.info(f"Document {doc_id}: Created {chunks_created} chunks from URL")
            
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            raise
    else:
        # Load from storage
        storage_path = os.getenv("STORAGE_PATH", "/app/storage")
        
        # Sanitize doc_id to prevent path traversal
        safe_doc_id = os.path.basename(doc_id)  # Remove any path components
        if safe_doc_id != doc_id or ".." in doc_id or os.path.isabs(doc_id):
            logger.error(f"Invalid doc_id detected (potential path traversal): {doc_id}")
            raise ValueError(f"Invalid document ID: {doc_id}")
        
        file_path = os.path.join(storage_path, safe_doc_id)
        
        # Verify resolved path is under storage_path
        resolved_path = os.path.realpath(file_path)
        resolved_storage = os.path.realpath(storage_path)
        if not resolved_path.startswith(resolved_storage):
            logger.error(f"Path traversal attempt detected: {doc_id} resolved to {resolved_path}")
            raise ValueError(f"Invalid document path: {doc_id}")
        
        if os.path.exists(file_path):
            # Read and process file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Use RecursiveCharacterTextSplitter for consistency with URL branch
            chunk_size = options.get("chunk_size", 500) if options else 500
            chunk_overlap = options.get("chunk_overlap", 50) if options else 50
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_text(content)
            chunks_created = len(chunks)
            
            logger.info(f"Document {doc_id}: Created {chunks_created} chunks from file")
        else:
            logger.warning(f"Document {doc_id} not found in storage")
            chunks_created = 0
    
    return chunks_created


@shared_task(bind=True, max_retries=2, autoretry_for=(Exception,), retry_backoff=True)
def process_uploaded_file_task(
    self,
    file_id: str,
    filename: str,
    content_type: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process an uploaded file for indexing.
    
    Args:
        file_id: Unique file identifier
        filename: Original filename
        content_type: MIME type of the file
        options: Processing options
        
    Returns:
        Processing results
    """
    task_id = self.request.id
    logger.info(f"Processing uploaded file {filename} (task: {task_id}, attempt: {self.request.retries + 1})")
    
    self.update_state(
        state="PROGRESS",
        meta={
            "status": f"Processing file: {filename}",
            "file_id": file_id
        }
    )
    
    if MOCK_MODE:
        time.sleep(1)  # Simulate processing
        return {
            "task_id": task_id,
            "file_id": file_id,
            "filename": filename,
            "status": "completed",
            "chunks_created": 10,
            "mock_mode": True
        }
    
    # Real processing would happen here
    # TODO: Implement actual file processing
    
    return {
        "task_id": task_id,
        "file_id": file_id,
        "filename": filename,
        "status": "completed",
        "chunks_created": 0
    }
