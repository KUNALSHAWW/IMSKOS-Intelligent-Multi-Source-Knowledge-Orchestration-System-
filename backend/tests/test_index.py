"""
IMSKOS Backend - Index API Tests
Tests for document indexing endpoints and crash prevention.
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Set mock mode for tests
os.environ["MOCK_MODE"] = "true"
os.environ["ENVIRONMENT"] = "test"

from app.main import create_app

app = create_app()
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_returns_200(self):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data
    
    def test_health_check_includes_mock_status(self):
        """Test that health endpoint includes mock mode info."""
        response = client.get("/health")
        data = response.json()
        
        assert "mock_mode" in data
        assert isinstance(data["mock_mode"], dict)


class TestIndexEndpoints:
    """Tests for /api/v1/index endpoints."""
    
    @patch("app.api.v1.index.get_celery_app")
    def test_start_indexing_returns_task_id(self, mock_celery):
        """Test that starting indexing returns a task ID immediately."""
        # Mock Celery task
        mock_task = MagicMock()
        mock_task.id = "test-task-123"
        mock_celery.return_value = (MagicMock(), MagicMock(delay=MagicMock(return_value=mock_task)))
        
        response = client.post(
            "/api/v1/index",
            json={"document_ids": ["doc-1", "doc-2"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
    
    @patch("app.api.v1.index.get_celery_app")
    def test_start_indexing_with_urls(self, mock_celery):
        """Test indexing with URLs."""
        mock_task = MagicMock()
        mock_task.id = "test-task-456"
        mock_celery.return_value = (MagicMock(), MagicMock(delay=MagicMock(return_value=mock_task)))
        
        response = client.post(
            "/api/v1/index",
            json={
                "document_ids": ["doc-1"],
                "urls": ["https://example.com/doc.html"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-456"
    
    def test_start_indexing_fallback_when_celery_unavailable(self):
        """Test that indexing returns mock response when Celery is unavailable."""
        with patch("app.api.v1.index.get_celery_app", return_value=(None, None)):
            response = client.post(
                "/api/v1/index",
                json={"document_ids": ["doc-1"]}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "mock" in data["task_id"]
            assert "MOCK" in data["message"]
    
    def test_get_task_status_for_mock_task(self):
        """Test getting status of a mock task."""
        response = client.get("/api/v1/index/status/mock-abc123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "mock-abc123"
        assert data["status"] == "SUCCESS"
    
    @patch("app.api.v1.index.get_celery_app")
    def test_get_task_status_pending(self, mock_celery):
        """Test getting status of pending task."""
        mock_result = MagicMock()
        mock_result.status = "PENDING"
        mock_task_class = MagicMock()
        mock_task_class.AsyncResult.return_value = mock_result
        mock_celery.return_value = (MagicMock(), mock_task_class)
        
        response = client.get("/api/v1/index/status/real-task-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PENDING"
    
    @patch("app.api.v1.index.get_celery_app")
    def test_get_task_status_progress(self, mock_celery):
        """Test getting status of in-progress task."""
        mock_result = MagicMock()
        mock_result.status = "PROGRESS"
        mock_result.info = {"current": 5, "total": 10, "status": "Processing..."}
        mock_task_class = MagicMock()
        mock_task_class.AsyncResult.return_value = mock_result
        mock_celery.return_value = (MagicMock(), mock_task_class)
        
        response = client.get("/api/v1/index/status/real-task-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PROGRESS"
        assert data["progress"]["current"] == 5
        assert data["progress"]["total"] == 10
    
    @patch("app.api.v1.index.get_celery_app")
    def test_get_task_status_success(self, mock_celery):
        """Test getting status of completed task."""
        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.result = {"processed": 10, "failed": 0, "chunks_created": 50}
        mock_task_class = MagicMock()
        mock_task_class.AsyncResult.return_value = mock_result
        mock_celery.return_value = (MagicMock(), mock_task_class)
        
        response = client.get("/api/v1/index/status/real-task-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["result"]["processed"] == 10
    
    @patch("app.api.v1.index.get_celery_app")
    def test_get_task_status_failure(self, mock_celery):
        """Test getting status of failed task."""
        mock_result = MagicMock()
        mock_result.status = "FAILURE"
        mock_result.result = Exception("Connection error")
        mock_task_class = MagicMock()
        mock_task_class.AsyncResult.return_value = mock_result
        mock_celery.return_value = (MagicMock(), mock_task_class)
        
        response = client.get("/api/v1/index/status/real-task-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FAILURE"
        assert "error" in data


class TestIndexingCrashPrevention:
    """Tests to verify indexing doesn't crash the application."""
    
    def test_indexing_returns_immediately(self):
        """Verify indexing endpoint returns quickly without blocking."""
        import time
        
        with patch("app.api.v1.index.get_celery_app", return_value=(None, None)):
            start = time.time()
            response = client.post(
                "/api/v1/index",
                json={"document_ids": [f"doc-{i}" for i in range(100)]}
            )
            elapsed = time.time() - start
            
            # Should return quickly without blocking on processing
            # Allow up to 5 seconds for CI environments with slower I/O
            assert elapsed < 5.0, f"Indexing took {elapsed}s - should return immediately"
            assert response.status_code == 200
    
    def test_app_remains_responsive_after_indexing_request(self):
        """Verify app remains responsive after indexing request."""
        with patch("app.api.v1.index.get_celery_app", return_value=(None, None)):
            # Start indexing
            response = client.post(
                "/api/v1/index",
                json={"document_ids": ["doc-1"]}
            )
            assert response.status_code == 200
            
            # App should still respond to health checks
            health_response = client.get("/health")
            assert health_response.status_code == 200
    
    def test_indexing_handles_missing_celery_gracefully(self):
        """Test that missing Celery doesn't crash the app."""
        with patch("app.api.v1.index.get_celery_app", return_value=(None, None)):
            response = client.post(
                "/api/v1/index",
                json={"document_ids": ["doc-1"]}
            )
            
            # Should return success with mock response, not crash
            assert response.status_code == 200
            assert "mock" in response.json()["task_id"].lower()
    
    @patch("app.api.v1.index.get_celery_app")
    def test_indexing_handles_celery_error_gracefully(self, mock_celery):
        """Test that Celery errors are handled without crashing - returns mock response."""
        mock_task_class = MagicMock()
        mock_task_class.delay.side_effect = Exception("Redis connection refused")
        mock_celery.return_value = (MagicMock(), mock_task_class)
        
        response = client.post(
            "/api/v1/index",
            json={"document_ids": ["doc-1"]}
        )
        
        # Should return success with mock response (graceful degradation), not crash
        assert response.status_code == 200
        data = response.json()
        assert "mock" in data["task_id"].lower()
        assert "mock" in data["message"].lower()


class TestIndexHealth:
    """Tests for indexing health endpoint."""
    
    def test_index_health_when_celery_unavailable(self):
        """Test index health endpoint when Celery is not available."""
        with patch("app.api.v1.index.get_celery_app", return_value=(None, None)):
            response = client.get("/api/v1/index/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["celery_available"] is False
    
    @patch("app.api.v1.index.get_celery_app")
    def test_index_health_with_celery(self, mock_celery):
        """Test index health endpoint with Celery available."""
        mock_app = MagicMock()
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {"worker1@localhost": []}
        mock_app.control.inspect.return_value = mock_inspect
        mock_celery.return_value = (mock_app, MagicMock())
        
        response = client.get("/api/v1/index/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["celery_available"] is True
        assert data["worker_count"] == 1


class TestQueryEndpoint:
    """Tests for query endpoint."""
    
    def test_query_returns_mock_response(self):
        """Test that query endpoint works in mock mode."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What is machine learning?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "response" in data
        # routing_reason may not be present in all mock responses
        if "routing_reason" in data:
            assert "mock" in data["routing_reason"].lower()
