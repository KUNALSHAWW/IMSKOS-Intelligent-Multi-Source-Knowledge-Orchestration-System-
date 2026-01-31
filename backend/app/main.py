"""
IMSKOS Backend - FastAPI Application Entry Point
Intelligent Multi-Source Knowledge Orchestration System API
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    settings = get_settings()
    
    # Startup
    logger.info(f"Starting IMSKOS API v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    
    # Log mock mode warnings
    for warning in settings.log_mock_mode_warnings():
        logger.warning(warning)
    
    mock_status = settings.get_mock_status()
    mock_services = [k for k, v in mock_status.items() if v]
    if mock_services:
        logger.info(f"Running in MOCK MODE for: {', '.join(mock_services)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down IMSKOS API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="IMSKOS API",
        description="""
## Intelligent Multi-Source Knowledge Orchestration System

IMSKOS is a production-quality RAG (Retrieval-Augmented Generation) system that combines:

- **Adaptive Query Routing**: LLM-powered routing to optimal data sources
- **Vector Store Retrieval**: DataStax Astra DB for semantic search
- **External Knowledge**: Wikipedia and web search integration
- **Streaming Responses**: Server-Sent Events for real-time token streaming

### Features

- 🔄 Intelligent query routing (Groq LLM with fallback heuristics)
- 🗄️ Scalable vector storage (Astra DB)
- 📚 Multi-source knowledge fusion
- ⚡ High-performance inference
- 📊 Query analytics and monitoring

### Mock Mode

When API keys are missing, the system runs in **MOCK MODE** with deterministic responses.
Check `/health` endpoint for mock mode status.
        """,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Next.js dev
            "http://localhost:8501",  # Streamlit
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router)
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
