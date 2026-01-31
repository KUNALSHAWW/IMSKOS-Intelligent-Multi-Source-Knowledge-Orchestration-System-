"""
IMSKOS Backend - Health Check Router
"""
from datetime import datetime

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the API and mock mode information."
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status, version, environment, and mock mode info.
    """
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        mock_mode=settings.get_mock_status(),
        timestamp=datetime.utcnow()
    )
