"""
IMSKOS Backend - API Tests
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Health check should return 200 status."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_returns_healthy_status(self, client):
        """Health check should return healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_check_includes_version(self, client):
        """Health check should include version info."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.1.0"
    
    def test_health_check_includes_mock_mode(self, client):
        """Health check should include mock mode status."""
        response = client.get("/health")
        data = response.json()
        assert "mock_mode" in data
        assert isinstance(data["mock_mode"], dict)


class TestQueryEndpoint:
    """Tests for POST /api/v1/query endpoint."""
    
    def test_query_returns_200(self, client):
        """Query should return 200 status."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What are the types of agent memory?"}
        )
        assert response.status_code == 200
    
    def test_query_returns_mock_response(self, client):
        """Query should return deterministic mock response in scaffold mode."""
        response = client.post(
            "/api/v1/query",
            json={"query": "Test query"}
        )
        data = response.json()
        
        # Check response structure
        assert "id" in data
        assert "response" in data
        assert "sources" in data
        assert "metrics" in data
        assert "routing_reason" in data
        
        # Check mock response content
        assert data["response"] == "This is a deterministic mock answer from IMSKOS scaffold."
        assert "mock" in data["routing_reason"].lower()
    
    def test_query_with_options(self, client):
        """Query should accept options."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Test query",
                "user_id": "user-123",
                "source": "vector",
                "options": {
                    "top_k": 10,
                    "temperature": 0.5,
                    "hyde": True
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "vector" in data["routing_reason"]
    
    def test_query_validates_empty_query(self, client):
        """Query should reject empty query string."""
        response = client.post(
            "/api/v1/query",
            json={"query": ""}
        )
        assert response.status_code == 422
    
    def test_query_sources_structure(self, client):
        """Query sources should have correct structure."""
        response = client.post(
            "/api/v1/query",
            json={"query": "Test query"}
        )
        data = response.json()
        
        assert len(data["sources"]) > 0
        source = data["sources"][0]
        
        assert "source_id" in source
        assert "similarity_score" in source
        assert "snippet" in source
        assert 0 <= source["similarity_score"] <= 1
    
    def test_query_metrics_structure(self, client):
        """Query metrics should have correct structure."""
        response = client.post(
            "/api/v1/query",
            json={"query": "Test query"}
        )
        data = response.json()
        metrics = data["metrics"]
        
        assert "elapsed_ms" in metrics
        assert "tokens" in metrics
        assert metrics["elapsed_ms"] > 0
        assert metrics["tokens"] > 0


class TestOpenAPI:
    """Tests for OpenAPI documentation."""
    
    def test_openapi_available(self, client):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
    
    def test_docs_available(self, client):
        """Swagger docs should be available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_available(self, client):
        """ReDoc should be available."""
        response = client.get("/redoc")
        assert response.status_code == 200
