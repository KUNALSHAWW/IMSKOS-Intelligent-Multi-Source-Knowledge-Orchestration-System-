"""
IMSKOS Backend - Celery Worker Configuration
Background task processing for document indexing and heavy operations.
"""
import os
import logging
from urllib.parse import urlparse

from celery import Celery, Task

logger = logging.getLogger(__name__)

# Redis URL from environment or default to local redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# Custom Task base class with default retry behavior
class BaseTask(Task):
    """Base task class with default retry configuration."""
    max_retries = 3
    default_retry_delay = 60


# Create Celery app
celery_app = Celery(
    "imskos_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker.tasks"],
    task_cls=BaseTask  # Use custom base task
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Task tracking
    task_track_started=True,
    task_acks_late=True,
    
    # Timeouts
    task_time_limit=600,  # 10 minutes hard limit
    task_soft_time_limit=540,  # 9 minutes soft limit (warning)
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for heavy tasks
    worker_concurrency=2,  # 2 concurrent workers
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    
    # Retry settings via task annotations (task_max_retries is not valid)
    task_default_retry_delay=60,  # 60 seconds between retries
    task_annotations={
        '*': {'max_retries': 3, 'default_retry_delay': 60}
    },
)

# Log sanitized Redis URL (mask credentials)
def _sanitize_redis_url(url: str) -> str:
    """Remove password from Redis URL for safe logging."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            # Replace password with ***
            safe_url = url.replace(f":{parsed.password}@", ":***@")
            return safe_url
        return url
    except Exception:
        return "redis://***"

logger.info(f"Celery app configured with broker: {_sanitize_redis_url(REDIS_URL)}")

__all__ = ["celery_app", "BaseTask"]
