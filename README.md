

<div align="center">

# 🧠 IMSKOS

### Intelligent Multi-Source Knowledge Orchestration System

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/KUNALSHAWW/IMSKOS)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.dev.yml)

*Production-quality RAG system with adaptive query routing, vector store retrieval, and multi-source knowledge fusion.*

[Quick Start](#-quick-start) • [Architecture](#-system-architecture) • [API Reference](#-api-reference) • [Contributing](#-contributing)

</div>

---

## 🎯 Project Overview

**IMSKOS v1.1** is a production-ready RAG (Retrieval-Augmented Generation) system featuring:

- **🔄 Adaptive Query Routing**: LLM-powered routing to optimal data sources (Groq with fallback heuristics)
- **🗄️ Distributed Vector Storage**: Scalable DataStax Astra DB for production-grade vector operations
- **⚡ High-Performance Inference**: Groq's lightning-fast LLM API for sub-second responses
- **🔗 Stateful Workflows**: LangGraph for complex, multi-step retrieval orchestration
- **🎨 Modern UI/UX**: Next.js + TypeScript frontend with shadcn/ui components
- **🛡️ Production Ready**: Docker, CI/CD, comprehensive testing, graceful mock mode

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IMSKOS v1.1 Architecture                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                           Frontend (Next.js + TypeScript)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Query Page  │  │Index Page   │  │ Docs Page   │  │  Analytics Page     │  │
│  │ - Prompt    │  │ - Upload    │  │ - List      │  │  - Charts           │  │
│  │ - Streaming │  │ - Progress  │  │ - Delete    │  │  - Metrics          │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │ HTTP/SSE
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Backend (FastAPI + Python)                          │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         API Layer (/api/v1)                             │  │
│  │  POST /query  │  POST /upload  │  GET /docs  │  POST /feedback        │  │
│  └─────────────────────────────────┬──────────────────────────────────────┘  │
│                                    │                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────┐  │
│  │  Query Router      │  │  Indexing Pipeline │  │  Auth & Sessions       │  │
│  │  - Groq LLM        │  │  - PDF/DOCX/Images │  │  - Supabase Auth       │  │
│  │  - Fallback Rules  │  │  - Chunking        │  │  - JWT Validation      │  │
│  │  - HyDE/Multi-Q    │  │  - Embeddings      │  │  - RBAC                │  │
│  └─────────┬──────────┘  └─────────┬──────────┘  └────────────────────────┘  │
│            │                       │                                          │
└────────────┼───────────────────────┼──────────────────────────────────────────┘
             │                       │
             ▼                       ▼
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────┐
│   Vector Store         │  │   Job Queue            │  │   Storage          │
│   (Astra DB)           │  │   (Redis + RQ)         │  │   (Supabase/S3)    │
│   - Semantic Search    │  │   - Async Processing   │  │   - File Storage   │
│   - Embeddings         │  │   - Progress Tracking  │  │   - Metadata       │
└────────────────────────┘  └────────────────────────┘  └────────────────────┘
```

---

## 📁 Folder Structure

```
IMSKOS/
├── frontend/                    # Next.js + TypeScript application
│   ├── src/
│   │   ├── app/                 # App Router pages
│   │   ├── components/          # React components
│   │   │   └── ui/              # shadcn/ui components
│   │   ├── lib/                 # Utilities and API client
│   │   ├── types/               # TypeScript types
│   │   └── test/                # Vitest tests
│   ├── Dockerfile
│   ├── package.json
│   └── tsconfig.json
│
├── backend/                     # FastAPI + Python application
│   ├── app/
│   │   ├── api/                 # API routes
│   │   │   └── v1/              # Version 1 endpoints
│   │   ├── core/                # Config, settings
│   │   └── models/              # Pydantic schemas
│   ├── tests/                   # pytest tests
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.dev.yml       # Development environment
├── .env.example                 # Environment variables template
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose (recommended)
- OR: Node.js 18+ and Python 3.10+

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/KUNALSHAWW/IMSKOS-Intelligent-Multi-Source-Knowledge-Orchestration-System-.git
cd IMSKOS-Intelligent-Multi-Source-Knowledge-Orchestration-System-

# 2. Copy environment file
cp .env.example .env

# 3. Start all services (runs in MOCK MODE if keys missing)
docker-compose -f docker-compose.dev.yml up --build

# 4. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

### Option 2: Local Development

**Backend:**
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 🔌 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Execute intelligent query |
| `POST` | `/api/v1/upload` | Upload file for indexing |
| `POST` | `/api/v1/index` | Trigger indexing job |
| `GET` | `/api/v1/index/{job_id}` | Get indexing job status |
| `GET` | `/api/v1/docs` | List indexed documents |
| `DELETE` | `/api/v1/docs/{doc_id}` | Delete document (admin) |
| `POST` | `/api/v1/feedback` | Submit query feedback |
| `GET` | `/health` | Health check |

### Query Request Example

```json
POST /api/v1/query
{
  "query": "What are the types of agent memory?",
  "user_id": "user-123",
  "source": "auto",
  "options": {
    "top_k": 5,
    "temperature": 0.0,
    "hyde": false
  }
}
```

### Query Response Example

```json
{
  "id": "query-abc123",
  "response": "Agent memory can be categorized into...",
  "sources": [
    {
      "source_id": "doc-1",
      "similarity_score": 0.92,
      "snippet": "Agent memory systems include...",
      "url": "/storage/docs/doc1.pdf"
    }
  ],
  "metrics": {
    "elapsed_ms": 245,
    "tokens": 156,
    "embedding_ms": 45,
    "retrieval_ms": 120,
    "llm_ms": 80
  },
  "routing_reason": "Routed to vector store: query contains technical AI terminology"
}
```

---

## 🔄 Mock Mode

IMSKOS runs in **MOCK MODE** when API keys are missing, enabling local development without external services.

| Missing Variable | Mock Behavior |
|------------------|---------------|
| `GROQ_API_KEY` | Deterministic heuristic routing |
| `ASTRA_DB_*` | In-memory vector store |
| `SUPABASE_*` | Local filesystem + SQLite |
| `HUGGINGFACE_API_KEY` | Local sentence-transformers |

Check `/health` endpoint to see mock mode status:

```json
{
  "status": "healthy",
  "version": "1.1.0",
  "mock_mode": {
    "supabase": true,
    "astra_db": true,
    "groq": true
  }
}
```

---

## ⚙️ Environment Variables

See [.env.example](.env.example) for the complete list. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | No* | Groq LLM API key |
| `ASTRA_DB_APPLICATION_TOKEN` | No* | Astra DB token |
| `ASTRA_DB_ID` | No* | Astra DB database ID |
| `SUPABASE_URL` | No* | Supabase project URL |
| `SUPABASE_ANON_KEY` | No* | Supabase anonymous key |
| `JWT_SECRET` | Yes | JWT signing secret |

*\* System runs in MOCK MODE if missing*

---

## 🧪 Testing

**Backend:**
```bash
cd backend
pytest -v
```

**Frontend:**
```bash
cd frontend
npm test          # Vitest unit tests
npm run test:e2e  # Playwright E2E tests
```

---

## 📈 Roadmap

### v1.1 (Current)
- [x] Next.js + TypeScript frontend
- [x] FastAPI backend with mock mode
- [x] Docker development environment
- [ ] Document upload & indexing
- [ ] SSE streaming responses
- [ ] Supabase auth integration

### v2.0 (Planned)
- [ ] Multi-modal support (images, PDFs)
- [ ] Advanced RAG (HyDE, Multi-Query)
- [ ] Graph RAG integration
- [ ] Real-time collaboration
- [ ] Public API endpoints

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/KUNALSHAWW/IMSKOS/issues)
- **Email**: kunalshawkol17@gmail.com
- **LinkedIn**: [Kunal Kumar Shaw](https://www.linkedin.com/in/kunal-kumar-shaw-443999205/)

---

<div align="center">

**Built with ❤️ using Next.js, FastAPI, LangGraph, Astra DB, and Groq**

*Elevating Information Retrieval to Intelligence*

</div>



