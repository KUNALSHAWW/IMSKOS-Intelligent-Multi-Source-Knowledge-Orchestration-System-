"""
IMSKOS E2E Smoke Tests
Tests for end-to-end indexing flow without crashing.

Run with:
    pytest tests/e2e/test_indexing_smoke.py -v
    
Requires services to be running:
    docker-compose -f docker-compose.dev.yml up -d
"""
import os
import pytest
import requests
import time

# Configuration from environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
STREAMLIT_URL = os.getenv("STREAMLIT_URL", "http://localhost:8501")
TIMEOUT = 30


@pytest.fixture(scope="module")
def check_backend():
    """Verify backend is running before tests."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip(f"Backend not healthy: {response.status_code}")
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Backend not available at {BACKEND_URL}: {e}")


@pytest.fixture(scope="module")
def check_streamlit():
    """Verify Streamlit is running before tests."""
    try:
        response = requests.get(f"{STREAMLIT_URL}/_stcore/health", timeout=5)
        if response.status_code != 200:
            pytest.skip(f"Streamlit not healthy: {response.status_code}")
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Streamlit not available at {STREAMLIT_URL}: {e}")


class TestIndexingE2E:
    """End-to-end tests for the indexing flow."""
    
    def test_backend_health(self, check_backend):
        """Test that backend is healthy."""
        response = requests.get(f"{BACKEND_URL}/health", timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_start_indexing_returns_immediately(self, check_backend):
        """
        Test that POST /api/v1/index returns immediately.
        This is the key fix for the Streamlit crash issue.
        """
        start = time.time()
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/index",
            json={"document_ids": [f"test-doc-{i}" for i in range(10)]},
            timeout=TIMEOUT
        )
        
        elapsed = time.time() - start
        
        # CRITICAL: Must return in under 5 seconds (should be <1s)
        # If this takes longer, it means indexing is blocking
        assert elapsed < 5.0, f"Indexing blocked for {elapsed}s - should return immediately!"
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
    
    def test_poll_task_status(self, check_backend):
        """Test that task status can be polled."""
        # Start indexing
        start_response = requests.post(
            f"{BACKEND_URL}/api/v1/index",
            json={"document_ids": ["poll-test-doc"]},
            timeout=TIMEOUT
        )
        
        assert start_response.status_code == 200
        task_id = start_response.json()["task_id"]
        
        # Poll for status
        status_response = requests.get(
            f"{BACKEND_URL}/api/v1/index/status/{task_id}",
            timeout=TIMEOUT
        )
        
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["task_id"] == task_id
        assert "status" in data
    
    def test_full_indexing_lifecycle(self, check_backend):
        """
        Test complete indexing lifecycle:
        1. Start indexing
        2. Poll for progress
        3. Verify completion
        """
        # Step 1: Start indexing
        response = requests.post(
            f"{BACKEND_URL}/api/v1/index",
            json={"document_ids": ["lifecycle-doc-1", "lifecycle-doc-2"]},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        task_id = response.json()["task_id"]
        
        # Step 2: Poll for completion (max 2 minutes)
        max_polls = 24  # 24 * 5s = 2 minutes
        final_status = None
        
        for i in range(max_polls):
            status_response = requests.get(
                f"{BACKEND_URL}/api/v1/index/status/{task_id}",
                timeout=TIMEOUT
            )
            
            assert status_response.status_code == 200
            status_data = status_response.json()
            final_status = status_data["status"]
            
            if final_status in ["SUCCESS", "FAILURE"]:
                break
            
            time.sleep(5)
        
        # Step 3: Verify completion (PENDING is not acceptable after 2 minutes of polling)
        assert final_status in ["SUCCESS", "FAILURE"], \
            f"Task did not complete within 2 minutes, final status: {final_status}"
    
    def test_backend_remains_responsive_during_indexing(self, check_backend):
        """
        Test that backend remains responsive while indexing is in progress.
        This verifies the async pattern is working correctly.
        """
        # Start indexing with many documents
        requests.post(
            f"{BACKEND_URL}/api/v1/index",
            json={"document_ids": [f"responsive-test-{i}" for i in range(50)]},
            timeout=TIMEOUT
        )
        
        # Immediately check that backend is still responsive
        for _ in range(5):
            health_response = requests.get(
                f"{BACKEND_URL}/health",
                timeout=5
            )
            assert health_response.status_code == 200, \
                "Backend became unresponsive after indexing request!"
            time.sleep(0.5)
    
    def test_index_health_endpoint(self, check_backend):
        """Test the index health endpoint."""
        response = requests.get(
            f"{BACKEND_URL}/api/v1/index/health",
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "celery_available" in data


class TestStreamlitIntegration:
    """Tests for Streamlit integration."""
    
    def test_streamlit_health(self, check_streamlit):
        """Test that Streamlit is healthy."""
        response = requests.get(
            f"{STREAMLIT_URL}/_stcore/health",
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_streamlit_survives_indexing(self, check_backend, check_streamlit):
        """
        KEY TEST: Verify Streamlit doesn't crash when indexing starts.
        
        This is the primary regression test for the original bug.
        """
        # Trigger indexing via backend API
        requests.post(
            f"{BACKEND_URL}/api/v1/index",
            json={"document_ids": ["streamlit-crash-test"]},
            timeout=TIMEOUT
        )
        
        # Wait a moment for any crash to occur
        time.sleep(3)
        
        # Verify Streamlit is still alive
        try:
            response = requests.get(
                f"{STREAMLIT_URL}/_stcore/health",
                timeout=10
            )
            assert response.status_code == 200, \
                "Streamlit crashed after indexing was triggered!"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Streamlit crashed after indexing: {e}")


class TestMockMode:
    """Tests for mock mode functionality."""
    
    def test_mock_task_returns_success(self, check_backend):
        """Test that mock tasks return success status."""
        response = requests.get(
            f"{BACKEND_URL}/api/v1/index/status/mock-test-123",
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["result"]["mock_mode"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
