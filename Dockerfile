FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 STREAMLIT_SERVER_ADDRESS=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# CPU-only torch first so sentence-transformers does not pull the multi-gigabyte CUDA build
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e . && useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Streamlit UI by default. For the REST API: docker run ... python -m imskos serve --host 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.headless=true"]
