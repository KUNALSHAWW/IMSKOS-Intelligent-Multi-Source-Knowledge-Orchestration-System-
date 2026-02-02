

<div align="center">

# 🧠 IMSKOS

### Intelligent Multi-Source Knowledge Orchestration System

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/KUNALSHAWW/IMSKOS)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io)

*Production-quality Agentic RAG system with adaptive query routing, vector store retrieval, and multi-source knowledge fusion.*

[Live Demo](#-live-demo) • [Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [API Keys](#-api-keys-setup)

</div>

---

## 🎯 Overview

**IMSKOS v2.0** is an advanced Agentic RAG (Retrieval-Augmented Generation) system deployed on **Hugging Face Spaces** featuring:

- **🔄 Adaptive Query Routing**: LLM-powered intelligent routing to optimal data sources
- **🗄️ Distributed Vector Storage**: DataStax Astra DB for scalable semantic search
- **⚡ High-Performance Inference**: Groq's ultra-fast LLM API for sub-second responses
- **🔗 Stateful Workflows**: LangGraph for complex, multi-step retrieval orchestration
- **📖 Multi-Source Knowledge**: Wikipedia integration for general knowledge queries
- **🎨 Modern Glassmorphism UI**: Beautiful, responsive dark theme with animations

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Intelligent Routing** | LLM automatically routes queries to the best data source |
| 📚 **Knowledge Base Indexing** | Index any web content into vector database |
| 🔍 **Semantic Search** | Find relevant documents using embeddings |
| 📖 **Wikipedia Integration** | Fallback to Wikipedia for general knowledge |
| 📊 **Analytics Dashboard** | Track query patterns and system performance |
| 🎨 **Modern UI** | Stunning glassmorphism design with animations |

---

## 🖼️ Screenshots

<div align="center">

### Query Interface
*Screenshot Coming Soon* - AI-powered query interface with intelligent routing

### Knowledge Base Indexing
*Screenshot Coming Soon* - Document indexing with progress tracking

### Analytics Dashboard
*Screenshot Coming Soon* - Query analytics and system monitoring

</div>

---

## 🚀 Quick Start

### Deploy on Hugging Face Spaces

1. **Create a new Space** on [Hugging Face](https://huggingface.co/new-space)
2. Select **Streamlit** as the SDK
3. Clone this repository or upload files
4. Add your secrets (see [API Keys Setup](#-api-keys-setup))
5. Your app will be live!

### Local Development

```bash
# Clone the repository
git clone https://github.com/KUNALSHAWW/IMSKOS.git
cd IMSKOS

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the app
streamlit run app.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         IMSKOS v2.0 Architecture                         │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌───────────────┐
                              │   User Query  │
                              └───────┬───────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend (UI)                           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ KB Indexing   │  │ Query Engine  │  │  Analytics    │                │
│  │ - URL Input   │  │ - Smart Route │  │  - Metrics    │                │
│  │ - Chunking    │  │ - Response    │  │  - History    │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     LangGraph Workflow Orchestration                     │
│                                                                          │
│   ┌─────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│   │  START  │───▶│ Intelligent      │───▶│ Route Decision          │   │
│   └─────────┘    │ Router (Groq)    │    │                         │   │
│                  └──────────────────┘    │  ┌─────────────────────┐│   │
│                                          │  │ vectorstore         ││   │
│                                          │  │ wiki_search         ││   │
│                                          │  └─────────────────────┘│   │
│                                          └─────────────────────────┘   │
│                                                    │                    │
│                          ┌─────────────────────────┼─────────────────┐  │
│                          │                         │                 │  │
│                          ▼                         ▼                 │  │
│                 ┌──────────────────┐     ┌──────────────────┐       │  │
│                 │  Vector Store    │     │  Wikipedia       │       │  │
│                 │  Retrieval       │     │  Search          │       │  │
│                 │  (Astra DB)      │     │  (API)           │       │  │
│                 └────────┬─────────┘     └────────┬─────────┘       │  │
│                          │                         │                 │  │
│                          └─────────────┬───────────┘                 │  │
│                                        ▼                             │  │
│                            ┌──────────────────┐                      │  │
│                            │  Generate        │                      │  │
│                            │  Response (LLM)  │                      │  │
│                            └────────┬─────────┘                      │  │
│                                     │                                │  │
│                                     ▼                                │  │
│                               ┌──────────┐                           │  │
│                               │   END    │                           │  │
│                               └──────────┘                           │  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         External Services                                │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  DataStax       │  │  Groq API       │  │  HuggingFace            │  │
│  │  Astra DB       │  │  (LLM)          │  │  (Embeddings)           │  │
│  │  - Vectors      │  │  - llama-3.1    │  │  - all-MiniLM-L6-v2     │  │
│  │  - Semantic     │  │  - Routing      │  │  - 384 dimensions       │  │
│  │    Search       │  │  - Generation   │  │                         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 API Keys Setup

### Required Services

| Service | Purpose | Get API Key |
|---------|---------|-------------|
| **DataStax Astra DB** | Vector database storage | [astra.datastax.com](https://astra.datastax.com) |
| **Groq** | LLM inference | [console.groq.com](https://console.groq.com) |

### Environment Variables

```env
# DataStax Astra DB
ASTRA_DB_APPLICATION_TOKEN=AstraCS:xxxxx
ASTRA_DB_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Groq LLM
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
```

### Hugging Face Spaces Secrets

1. Go to your Space **Settings**
2. Navigate to **Secrets** section
3. Add each variable:
   - `ASTRA_DB_APPLICATION_TOKEN`
   - `ASTRA_DB_ID`
   - `GROQ_API_KEY`

---

## 📁 Project Structure

```
IMSKOS/
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── .env.example                # Environment template
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit + Custom CSS |
| **Orchestration** | LangGraph |
| **LLM** | Groq (llama-3.1-8b-instant) |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Vector DB** | DataStax Astra DB (Cassandra) |
| **External KB** | Wikipedia API |

---

## 🎨 UI Features

- **Glassmorphism Design**: Modern frosted glass aesthetic
- **Animated Gradients**: Dynamic color transitions
- **Responsive Layout**: Works on all screen sizes
- **Dark Theme**: Easy on the eyes
- **Real-time Feedback**: Loading animations and progress indicators

---

## 📊 How It Works

### 1. Knowledge Base Indexing
- Provide URLs to index
- Documents are fetched and chunked (500 tokens, 50 overlap)
- Embeddings generated using HuggingFace
- Vectors stored in Astra DB

### 2. Intelligent Query Routing
- User submits a question
- Groq LLM analyzes the query
- Routes to **Vector Store** (indexed content) or **Wikipedia** (general knowledge)

### 3. Response Generation
- Retrieved documents provide context
- LLM generates comprehensive answer
- Sources displayed for transparency

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Kunal Shaw**

- GitHub: [@KUNALSHAWW](https://github.com/KUNALSHAWW)
- LinkedIn: [Kunal Shaw](https://linkedin.com/in/kunalshaw)
- Email: kunalshawkol17@gmail.com

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ using LangGraph, Astra DB, and Groq

</div>



