# PR: fix/hotfix-streamlit-index-crash

## 🔥 Hotfix: Prevent Streamlit Crash on Index Documents Click

### Summary
This PR fixes a critical bug where clicking "Index Documents" in the Streamlit UI causes the Streamlit process to crash or hang. The fix implements defensive error handling, timeout protection, and mock mode support to ensure the UI always remains responsive.

### Problem
When users click "Index Documents" in the Streamlit UI:
1. The `initialize_cassandra()` function can hang indefinitely if Astra DB is hibernated or unreachable
2. The `load_and_process_documents()` function can raise unhandled exceptions
3. Long-running operations block the Streamlit main thread, causing the UI to become unresponsive
4. No graceful fallback exists when external services are unavailable

### Solution
This hotfix implements:

1. **Mock Mode Detection & Banner**
   - Automatically detects missing critical env vars (`ASTRA_DB_APPLICATION_TOKEN`, `ASTRA_DB_ID`, `GROQ_API_KEY`)
   - Shows clear warning banner when running in mock mode
   - Allows development/testing without external service dependencies

2. **Safe Wrapper Functions**
   - `safe_initialize_cassandra()` - wraps Cassandra init with try/except, returns error dict instead of raising
   - `safe_load_and_process_documents()` - handles document loading errors gracefully
   - `run_indexing_with_timeout()` - uses ThreadPoolExecutor with configurable timeout

3. **Emergency Kill Switch**
   - `ENABLE_INDEX_BACKGROUND=false` completely disables indexing
   - Useful for immediate hotfix deployment without code changes

4. **Configurable Timeouts**
   - `ASTRA_CONNECT_TIMEOUT` - configurable DB connection timeout (default 30s)
   - ThreadPoolExecutor timeout of 120s for embedding generation

### Files Changed

| File | Changes |
|------|---------|
| `app.py` | Added mock mode detection, safe wrapper functions, timeout protection, error handling |
| `.env.example` | Added `MOCK_MODE`, `ENABLE_INDEX_BACKGROUND`, `ASTRA_CONNECT_TIMEOUT`, `BACKEND_URL`, `REDIS_URL` |
| `tests/test_streamlit_hotfix.py` | New file with 20 test cases |

### New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOCK_MODE` | `false` | Enable mock mode for all external services |
| `ENABLE_INDEX_BACKGROUND` | `true` | Toggle to disable indexing completely (emergency kill switch) |
| `ASTRA_CONNECT_TIMEOUT` | `30` | Connection timeout in seconds |
| `BACKEND_URL` | `http://localhost:8000` | FastAPI backend URL for Streamlit |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for Celery workers |

### How to Run Locally

```bash
# 1. Clone and checkout the branch
git clone https://github.com/KUNALSHAWW/IMSKOS-Intelligent-Multi-Source-Knowledge-Orchestration-System-.git
cd IMSKOS-Intelligent-Multi-Source-Knowledge-Orchestration-System-
git checkout fix/hotfix-streamlit-index-crash

# 2. Copy environment template and set mock mode
cp .env.example .env
echo "MOCK_MODE=true" >> .env

# 3. Run with Docker Compose (if available)
docker-compose -f docker-compose.dev.yml up --build

# OR run Streamlit directly
pip install -r requirements.txt
streamlit run app.py
```

### How to Test

```bash
# Run the hotfix unit tests
python -m pytest tests/test_streamlit_hotfix.py -v

# Expected output: 20 passed
```

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-9.0.2, pluggy-1.6.0
collected 20 items

tests/test_streamlit_hotfix.py::TestMockModeDetection::test_mock_mode_enabled_when_env_var_true PASSED
tests/test_streamlit_hotfix.py::TestMockModeDetection::test_mock_mode_disabled_when_env_var_false PASSED
tests/test_streamlit_hotfix.py::TestMockModeDetection::test_missing_env_vars_detected PASSED
tests/test_streamlit_hotfix.py::TestMockModeDetection::test_partial_missing_env_vars PASSED
tests/test_streamlit_hotfix.py::TestSafeInitializeCassandra::test_returns_mock_result_in_mock_mode PASSED
tests/test_streamlit_hotfix.py::TestSafeInitializeCassandra::test_catches_connection_timeout PASSED
tests/test_streamlit_hotfix.py::TestSafeInitializeCassandra::test_catches_connection_error PASSED
tests/test_streamlit_hotfix.py::TestSafeDocumentLoading::test_returns_mock_documents_in_mock_mode PASSED
tests/test_streamlit_hotfix.py::TestSafeDocumentLoading::test_catches_url_fetch_errors PASSED
tests/test_streamlit_hotfix.py::TestThreadPoolTimeout::test_timeout_returns_error_dict PASSED
tests/test_streamlit_hotfix.py::TestThreadPoolTimeout::test_successful_execution_within_timeout PASSED
tests/test_streamlit_hotfix.py::TestEnableIndexBackgroundToggle::test_indexing_disabled_when_toggle_false PASSED
tests/test_streamlit_hotfix.py::TestEnableIndexBackgroundToggle::test_indexing_enabled_by_default PASSED
tests/test_streamlit_hotfix.py::TestEnableIndexBackgroundToggle::test_indexing_enabled_when_toggle_true PASSED
tests/test_streamlit_hotfix.py::TestAstraConnectTimeout::test_custom_timeout_from_env PASSED
tests/test_streamlit_hotfix.py::TestAstraConnectTimeout::test_default_timeout_when_not_set PASSED
tests/test_streamlit_hotfix.py::TestBackendEnqueue::test_successful_enqueue_returns_task_id PASSED
tests/test_streamlit_hotfix.py::TestBackendEnqueue::test_backend_unavailable_returns_fallback PASSED
tests/test_streamlit_hotfix.py::TestMockTaskStatus::test_mock_task_id_returns_completed_status PASSED
tests/test_streamlit_hotfix.py::TestIndexingFlowMockMode::test_full_mock_indexing_flow PASSED

============================= 20 passed in 0.17s ==============================
```

### Reproduction Steps (Before Fix)

1. Clone repo and set up without API keys
2. Run `streamlit run app.py`
3. Click "Index Documents" button
4. **Before fix**: Streamlit crashes or hangs
5. **After fix**: Shows error message with recovery options, UI remains responsive

### Acceptance Criteria

- [x] Streamlit does not crash when clicking "Index Documents" without API keys
- [x] Mock mode banner appears when critical env vars are missing
- [x] `ENABLE_INDEX_BACKGROUND=false` disables indexing with clear message
- [x] Connection timeout of 30s (configurable) prevents indefinite hangs
- [x] All error conditions show user-friendly messages with recovery steps
- [x] Full traceback available in collapsible expander for debugging
- [x] All 20 unit tests pass
- [x] No breaking changes to existing functionality

### Rollback Steps

If this PR causes issues:

```bash
# 1. Revert to previous branch
git checkout feat/project-scaffold

# 2. Or revert the specific commit
git revert <commit-hash>

# 3. Or disable indexing without code change
echo "ENABLE_INDEX_BACKGROUND=false" >> .env
```

### Screenshots / Logs

#### Mock Mode Banner
When `MOCK_MODE=true` or critical env vars missing:
```
⚠️ MOCK MODE: Missing environment variables: ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ID, GROQ_API_KEY.
Set MOCK_MODE=true for explicit mock behavior, or provide the required keys.
```

#### Error Handling
When DB connection fails:
```
⚠️ Astra DB Connection Failed

[Error message]

💡 Solutions:
1. Go to astra.datastax.com
2. Check if your database is Active (not Hibernated)
3. If hibernated, click "Resume" and wait 1-2 minutes
4. Or set MOCK_MODE=true to test without DB

🔍 Error Details (Technical) [expandable traceback]
```

### Related Issues
- Fixes: Streamlit crash on Index Documents click

### Next Steps (Future PRs)
1. `feat/worker-indexing-celery` - Implement proper Celery background worker
2. Update Streamlit to poll task status instead of blocking
3. Add `/diagnostics` endpoint for health monitoring
4. Add E2E smoke tests to CI

---

**Note**: This is a minimal, safe, and reversible hotfix designed for immediate deployment. The longer-term solution with proper background workers will follow in subsequent PRs.
