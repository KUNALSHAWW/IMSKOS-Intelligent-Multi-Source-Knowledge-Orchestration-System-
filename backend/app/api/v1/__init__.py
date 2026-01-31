# API v1 module
from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.query import router as query_router

api_router = APIRouter()

# Include all v1 routers
api_router.include_router(health_router)
api_router.include_router(query_router)

__all__ = ["api_router"]
