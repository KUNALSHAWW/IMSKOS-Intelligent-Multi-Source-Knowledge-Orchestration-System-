# IMSKOS Development Guidelines

## Critical Rules for Agent/Copilot Assistance

These rules MUST be followed when implementing features to prevent crashes and maintain system stability.

---

## ⚠️ Rule 1: Never Run Blocking Operations on Web Thread

**Problem:** Heavy operations (document processing, embeddings, OCR) running on the main web thread will crash Streamlit and make FastAPI unresponsive.

**❌ WRONG:**
```python
@app.post("/api/index")
async def index_documents(request: IndexRequest):
    # BAD: This blocks the web server for minutes
    result = process_documents(request.files)  # Heavy I/O
    embeddings = generate_embeddings(result)    # CPU-intensive
    store_in_vector_db(embeddings)              # Network I/O
    return {"status": "done"}
```

**✅ CORRECT:**
```python
@app.post("/api/index")
async def index_documents(request: IndexRequest):
    # GOOD: Queue task and return immediately
    task = index_documents_task.delay(request.document_ids)
    return {"task_id": task.id, "status": "pending"}

@app.get("/api/index/status/{task_id}")
async def get_status(task_id: str):
    from celery.result import AsyncResult
    result = AsyncResult(task_id)
    return {"status": result.status, "result": result.result}
```

---

## ⚠️ Rule 2: Always Implement Worker Queue Pattern for Heavy Operations

**Required Pattern:**
1. API receives request → validates → enqueues task → returns task_id
2. Worker picks up task → processes → updates status
3. UI polls status endpoint → displays progress

**Implementation checklist:**
- [ ] Create Celery task in `backend/app/worker/tasks.py`
- [ ] Add API endpoint that returns task_id immediately
- [ ] Add status polling endpoint
- [ ] Update UI to poll instead of wait

---

## ⚠️ Rule 3: Never Remove UI Without Compatibility Layer

If migrating from one frontend to another (e.g., Streamlit → Next.js):
- [ ] Keep BOTH frontends running
- [ ] Add feature flag: `USE_LEGACY_UI=true`
- [ ] Add smoke tests for BOTH UIs before merging
- [ ] Document migration path

---

## ⚠️ Rule 4: All PRs Must Include

Before submitting a PR, ensure:
- [ ] Steps to run locally (exact commands)
- [ ] Required environment variables listed
- [ ] Reproducible test script
- [ ] Smoke test for affected flows
- [ ] Rollback instructions

---

## ⚠️ Rule 5: Error Handling Requirements

**Every external call MUST have:**
```python
import requests
import httpx

try:
    result = external_api_call(timeout=30)  # Always set timeout
except (TimeoutError, requests.exceptions.Timeout, httpx.TimeoutException) as e:
    logger.error("Timeout calling external API", extra={"endpoint": url, "error": str(e)})
    return fallback_response()
except (requests.exceptions.ConnectionError, httpx.ConnectError) as e:
    logger.error("Connection failed", extra={"endpoint": url, "error": str(e)})
    return fallback_response()
except Exception as e:
    logger.exception("Unexpected error")
    raise HTTPException(status_code=500, detail="Service temporarily unavailable")
```

---

## ⚠️ Rule 6: Timeout Every Blocking Call

**❌ Never do this:**
```python
response = requests.get(url)  # No timeout = infinite wait
```

**✅ Always do this:**
```python
response = requests.get(url, timeout=30)  # Explicit timeout
```

---

## ⚠️ Rule 7: Required Tests for Risky Changes

| Change Type | Required Tests |
|------------|----------------|
| New API endpoint | Unit test + integration test |
| Background task | Task execution test + timeout test |
| UI action | E2E smoke test |
| Docker change | Build test + health check test |

---

## Running Tests

```bash
# Unit tests
cd backend
pytest tests/ -v

# E2E smoke tests (requires running services)
docker-compose -f docker-compose.dev.yml up -d
pytest tests/e2e/ -v

# Quick health check
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

---

## Quick Start for Development

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Set MOCK_MODE for testing without external services
echo "MOCK_MODE=true" >> .env

# 3. Start all services
docker-compose -f docker-compose.dev.yml up --build

# 4. Verify services are running
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health

# 5. Test indexing doesn't crash
curl -X POST http://localhost:8000/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["test-doc"]}'
```

---

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   FastAPI       │────▶│   Redis         │
│   (UI)          │     │   (Backend)     │     │   (Queue)       │
│   :8501         │     │   :8000         │     │   :6379         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │   Celery        │
                                                │   (Worker)      │
                                                │   Heavy Tasks   │
                                                └─────────────────┘
```

**Key principle:** UI and API never do heavy processing directly. Everything goes through the task queue.
