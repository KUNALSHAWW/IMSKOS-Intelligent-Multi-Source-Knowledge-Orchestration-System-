"""
� IMSKOS - Intelligent Multi-Source Knowledge Orchestration System
===================================================================
🚀 Advanced Agentic RAG Framework with Dynamic Routing & Distributed Vector Storage

Deployed on Hugging Face Spaces - Enterprise-grade intelligent query routing system featuring:
• LangGraph for stateful workflow orchestration
• DataStax Astra DB for distributed vector storage  
• Groq LLM for high-performance inference
• Adaptive routing between proprietary knowledge base and Wikipedia
• Real-time semantic search with HuggingFace embeddings

Author: Kunal Shaw | GitHub: github.com/KUNALSHAWW
"""

# =============================================================================
# CRITICAL: Disable TensorFlow completely - we only need PyTorch for embeddings
# This MUST be set before ANY imports that might trigger TensorFlow loading
# =============================================================================
import os
import sys
import warnings
import logging

# Tell transformers/HuggingFace to use PyTorch only, not TensorFlow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Suppress TensorFlow C++ logging completely (in case it still gets loaded)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL only
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_SILENCE_DEPRECATION_WARNINGS"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

# Suppress all Python warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Pre-configure logging to suppress TensorFlow/Keras if they somehow get imported
for logger_name in ["tensorflow", "tf_keras", "keras", "h5py", "absl", "transformers"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

import streamlit as st
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import random
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Load environment variables from .env file
load_dotenv()

# Set USER_AGENT to suppress warnings from web loaders
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "IMSKOS/2.0 (Intelligent Multi-Source Knowledge Orchestration System)"

# =============================================================================
# HOTFIX: Mock Mode and Environment Guards
# =============================================================================
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
ENABLE_INDEX_BACKGROUND = os.getenv("ENABLE_INDEX_BACKGROUND", "true").lower() == "true"
ASTRA_CONNECT_TIMEOUT = int(os.getenv("ASTRA_CONNECT_TIMEOUT", "30"))

# Required environment variables for full functionality
REQUIRED_ENV_VARS = {
    "ASTRA_DB_APPLICATION_TOKEN": "DataStax Astra DB token for vector storage",
    "ASTRA_DB_ID": "DataStax Astra DB identifier",
    "GROQ_API_KEY": "Groq API key for LLM inference"
}

# Check missing env vars (will be used to show banner)
MISSING_ENV_VARS = []
for var, desc in REQUIRED_ENV_VARS.items():
    if not os.getenv(var):
        # Also check Streamlit secrets
        try:
            if not (hasattr(st, 'secrets') and var in st.secrets):
                MISSING_ENV_VARS.append(var)
        except Exception:
            MISSING_ENV_VARS.append(var)

# Compatibility shim for different typing.ForwardRef._evaluate signatures
try:
    from typing import ForwardRef as _ForwardRef

    _orig_forwardref_evaluate = getattr(_ForwardRef, "_evaluate", None)
    if _orig_forwardref_evaluate is not None:
        def _evaluate_compat(self, globalns, localns, *args, **kwargs):
            try:
                return _orig_forwardref_evaluate(self, globalns, localns, *args, **kwargs)
            except TypeError:
                recursive_guard = args[0] if args else set()
                return _orig_forwardref_evaluate(self, globalns, localns, recursive_guard=recursive_guard)
        _ForwardRef._evaluate = _evaluate_compat
except Exception:
    pass

import cassio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Literal
import time
import json
from datetime import datetime
import traceback
import requests

# Track if hotfix warnings have been shown this session
_HOTFIX_BANNER_SHOWN = False

# Page Configuration
st.set_page_config(
    page_title="IMSKOS | AI Knowledge Orchestrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/KUNALSHAWW/IMSKOS',
        'Report a bug': 'https://github.com/KUNALSHAWW/IMSKOS/issues',
        'About': '# IMSKOS v2.0\nIntelligent Multi-Source Knowledge Orchestration System'
    }
)

# ==================== CUSTOM CSS - Modern Glassmorphism Design ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        position: relative;
    }
    
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 8s ease infinite;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 80px rgba(102, 126, 234, 0.5);
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    .hero-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 50px;
        color: #a78bfa;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .glass-card-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .glass-card-content {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Feature Pills */
    .feature-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 50px;
        color: #c4b5fd;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
        transition: all 0.2s ease;
    }
    
    .feature-pill:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        transform: scale(1.05);
    }
    
    /* Metric Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Route Indicators */
    .route-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        animation: pulse-glow 2s infinite;
    }
    
    .route-vector {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.3) 100%);
        border: 1px solid rgba(59, 130, 246, 0.5);
        color: #93c5fd;
    }
    
    .route-wiki {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.3) 100%);
        border: 1px solid rgba(245, 158, 11, 0.5);
        color: #fcd34d;
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.5); }
    }
    
    /* Response Box */
    .response-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 1.5rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .response-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
    }
    
    .response-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #6ee7b7;
        margin-bottom: 1rem;
    }
    
    .response-text {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        line-height: 1.8;
    }
    
    /* Source Documents */
    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.1) 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.85);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.1) 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.85);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-left: 4px solid #10b981;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.85);
    }
    
    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Text Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 0.5rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(255, 255, 255, 0.6) !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%) !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Animations */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Code Blocks */
    code {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #c4b5fd !important;
        border-radius: 6px !important;
        padding: 0.2rem 0.5rem !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: rgba(255, 255, 255, 0.5);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }
    
    .footer a {
        color: #a78bfa;
        text-decoration: none;
    }
    
    .footer a:hover {
        color: #c4b5fd;
    }
    
    /* Particles Background Effect */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        overflow: hidden;
        z-index: -1;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Configuration & Initialization ====================

# Inspirational quotes for loading screens
LOADING_QUOTES = [
    "🧠 Training neural pathways...",
    "🔮 Consulting the knowledge oracle...",
    "⚡ Charging the vector capacitors...",
    "🌌 Traversing the semantic space...",
    "🎯 Calibrating intelligent routing...",
    "🚀 Initializing quantum processors...",
    "💫 Awakening the AI consciousness...",
]

class Config:
    """Centralized configuration management"""
    
    @staticmethod
    def load_env_variables():
        """Load and validate environment variables from multiple sources
        
        Priority order:
        1. Streamlit secrets (for Streamlit Cloud / HuggingFace Spaces)
        2. Environment variables (for local development / Docker)
        """
        
        def get_secret(key: str) -> Optional[str]:
            """Get secret from Streamlit secrets or environment variables"""
            # First check Streamlit secrets (works on HuggingFace Spaces)
            try:
                if hasattr(st, 'secrets') and key in st.secrets:
                    return st.secrets[key]
            except Exception:
                pass
            # Fall back to environment variables
            return os.getenv(key)
        
        required_vars = {
            "ASTRA_DB_APPLICATION_TOKEN": get_secret("ASTRA_DB_APPLICATION_TOKEN"),
            "ASTRA_DB_ID": get_secret("ASTRA_DB_ID"),
            "GROQ_API_KEY": get_secret("GROQ_API_KEY")
        }
        
        missing_vars = [key for key, value in required_vars.items() if not value]
        
        if missing_vars:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ Configuration Required</h4>
                <p>Missing environment variables: <code>{', '.join(missing_vars)}</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📖 Setup Instructions", expanded=True):
                st.markdown("""
                ### 🔐 Required API Keys
                
                **1. DataStax Astra DB** (Vector Storage)
                - Sign up at [astra.datastax.com](https://astra.datastax.com)
                - Create a Serverless Vector Database
                - Get your `ASTRA_DB_APPLICATION_TOKEN` and `ASTRA_DB_ID`
                
                **2. Groq API** (LLM Inference)
                - Sign up at [console.groq.com](https://console.groq.com)
                - Generate an API key
                
                ### 🚀 Hugging Face Spaces Setup
                1. Go to your Space Settings
                2. Navigate to "Secrets" section
                3. Add the following secrets:
                   - `ASTRA_DB_APPLICATION_TOKEN`
                   - `ASTRA_DB_ID`
                   - `GROQ_API_KEY`
                
                ### 💻 Local Development
                Create a `.env` file in the project root:
                ```
                ASTRA_DB_APPLICATION_TOKEN=your_token_here
                ASTRA_DB_ID=your_db_id_here
                GROQ_API_KEY=your_groq_key_here
                ```
                """)
            st.stop()
        
        return required_vars
    
    @staticmethod
    def get_default_urls():
        """Default knowledge base URLs for AI/ML topics"""
        return [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

# ==================== State Management Classes ====================

class RouteQuery(BaseModel):
    """Pydantic model for query routing decisions"""
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Route query to wikipedia or vectorstore based on content",
    )

class GraphState(TypedDict):
    """LangGraph state schema"""
    question: str
    generation: str
    documents: List[str]
    route: str

# =============================================================================
# HOTFIX: Safe indexing functions with error handling
# =============================================================================
def show_mock_mode_banner():
    """Display mock mode banner if critical env vars are missing."""
    global _HOTFIX_BANNER_SHOWN
    if MISSING_ENV_VARS and not _HOTFIX_BANNER_SHOWN:
        st.warning(
            f"⚠️ **MOCK MODE**: Missing environment variables: {', '.join(MISSING_ENV_VARS)}. "
            f"Set `MOCK_MODE=true` for explicit mock behavior, or provide the required keys."
        )
        _HOTFIX_BANNER_SHOWN = True
        return True
    if MOCK_MODE:
        st.info("🔧 Running in **MOCK MODE** - external services are mocked.")
        return True
    return False


def safe_initialize_cassandra(kb_manager, timeout_seconds: int = None) -> dict:
    """
    HOTFIX: Safely initialize Cassandra with try/except and timeout.
    Returns dict with 'success' boolean and 'error' message if failed.
    Does not raise exceptions - keeps UI responsive.
    """
    timeout = timeout_seconds or ASTRA_CONNECT_TIMEOUT
    
    # Check if in mock mode
    if MOCK_MODE or MISSING_ENV_VARS:
        logging.info("HOTFIX: Skipping Cassandra init in mock mode")
        return {"success": True, "mock": True, "message": "Mock mode - Cassandra init skipped"}
    
    try:
        with st.spinner("Initializing vector DB connection..."):
            kb_manager.initialize_cassandra(timeout_seconds=timeout)
        return {"success": True, "mock": False}
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.exception("HOTFIX: Failed to initialize Cassandra/Astra client")
        
        # Return error info without raising
        return {
            "success": False,
            "mock": False,
            "error": str(e),
            "traceback": error_trace
        }


def safe_load_and_process_documents(kb_manager, urls: List[str], progress_callback=None) -> dict:
    """
    HOTFIX: Safely load and process documents with try/except.
    Returns dict with 'success', 'doc_splits' or 'error'.
    """
    # Check if in mock mode
    if MOCK_MODE:
        logging.info("HOTFIX: Mock document loading")
        # Return mock documents
        mock_docs = [
            Document(page_content=f"[MOCK] Sample document chunk {i} from URL indexing.", metadata={"source": "mock"})
            for i in range(3)
        ]
        return {"success": True, "mock": True, "doc_splits": mock_docs}
    
    try:
        doc_splits = kb_manager.load_and_process_documents(urls, progress_callback)
        return {"success": True, "mock": False, "doc_splits": doc_splits}
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.exception("HOTFIX: Document loading/processing failed")
        return {
            "success": False,
            "mock": False,
            "error": str(e),
            "traceback": error_trace
        }


def run_indexing_with_timeout(func, *args, timeout_seconds: int = 30, **kwargs) -> dict:
    """
    HOTFIX: Run a function in a thread with timeout.
    Prevents Streamlit main thread from blocking indefinitely.
    Returns dict with 'success', 'result' or 'error'/'timeout'.
    """
    _executor = ThreadPoolExecutor(max_workers=1)
    
    try:
        future = _executor.submit(func, *args, **kwargs)
        result = future.result(timeout=timeout_seconds)
        return {"success": True, "result": result, "timeout": False}
    except FuturesTimeoutError:
        logging.warning(f"HOTFIX: Operation timed out after {timeout_seconds}s")
        return {
            "success": False,
            "timeout": True,
            "error": f"Operation timed out after {timeout_seconds} seconds"
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.exception("HOTFIX: Threaded operation failed")
        return {
            "success": False,
            "timeout": False,
            "error": str(e),
            "traceback": error_trace
        }
    finally:
        _executor.shutdown(wait=False)


def enqueue_indexing_to_backend(document_ids: List[str], urls: List[str] = None) -> dict:
    """
    HOTFIX: Enqueue indexing task to FastAPI backend worker.
    Returns dict with 'success', 'task_id' or 'error'.
    Falls back gracefully if backend unavailable.
    """
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    try:
        payload = {"document_ids": document_ids}
        if urls:
            payload["urls"] = urls
        
        response = requests.post(
            f"{backend_url}/api/v1/index",
            json=payload,
            timeout=10  # Short timeout for enqueue
        )
        
        if response.status_code == 200:
            data = response.json()
            return {"success": True, "task_id": data.get("task_id"), "data": data}
        else:
            return {
                "success": False,
                "error": f"Backend returned status {response.status_code}",
                "response": response.text
            }
    except requests.exceptions.Timeout:
        logging.warning("HOTFIX: Backend enqueue request timed out")
        return {"success": False, "error": "Backend connection timed out", "fallback": True}
    except requests.exceptions.ConnectionError:
        logging.warning("HOTFIX: Could not connect to backend")
        return {"success": False, "error": "Backend unavailable", "fallback": True}
    except Exception as e:
        logging.exception("HOTFIX: Backend enqueue failed")
        return {"success": False, "error": str(e), "fallback": True}


def poll_indexing_status(task_id: str) -> dict:
    """
    HOTFIX: Poll backend for indexing task status.
    Returns dict with task state and progress info.
    """
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    # Handle mock task IDs
    if task_id.startswith("mock-"):
        return {
            "success": True,
            "state": "SUCCESS",
            "mock": True,
            "progress": {"current": 100, "total": 100, "status": "Mock indexing complete"}
        }
    
    try:
        response = requests.get(
            f"{backend_url}/api/v1/index/status/{task_id}",
            timeout=5
        )
        
        if response.status_code == 200:
            return {"success": True, **response.json()}
        else:
            return {"success": False, "error": f"Status check returned {response.status_code}"}
    except Exception as e:
        logging.exception("HOTFIX: Status poll failed")
        return {"success": False, "error": str(e)}


# ==================== Core System Classes ====================

class KnowledgeBaseManager:
    """Manages document ingestion and vector storage with lazy initialization"""
    
    def __init__(self, astra_token: str, astra_db_id: str):
        self.astra_token = astra_token
        self.astra_db_id = astra_db_id
        self.embeddings = None
        self.vector_store = None
        self._cassandra_initialized = False
        
    def initialize_cassandra(self, timeout_seconds: int = None):
        """Initialize Cassandra connection with timeout (lazy - only when needed)"""
        timeout = timeout_seconds or ASTRA_CONNECT_TIMEOUT
        
        if not self._cassandra_initialized:
            def _connect():
                cassio.init(token=self.astra_token, database_id=self.astra_db_id)
            
            try:
                # Run connection in a thread with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_connect)
                    future.result(timeout=timeout)  # Wait with timeout
                self._cassandra_initialized = True
            except FuturesTimeoutError:
                raise ConnectionError(
                    f"Connection to Astra DB timed out after {timeout} seconds.\n"
                    f"Your database may be hibernated. Please:\n"
                    f"1. Go to astra.datastax.com\n"
                    f"2. Check if your database is Active\n"
                    f"3. If Hibernated, click 'Resume' and wait 1-2 minutes\n"
                    f"4. Try indexing again"
                )
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to Astra DB. Please check:\n"
                    f"1. Your database is active (not hibernated) at astra.datastax.com\n"
                    f"2. Your ASTRA_DB_APPLICATION_TOKEN is valid\n"
                    f"3. Your ASTRA_DB_ID is correct\n\n"
                    f"Error: {str(e)}"
                )
        
    def setup_embeddings(self):
        """Initialize HuggingFace embeddings with robust error handling"""
        import torch
        
        # Force CPU to avoid meta tensor issues with GPU
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        try:
            # Try with explicit CPU device to avoid meta tensor issues
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        except Exception as e:
            error_msg = str(e)
            if "meta tensor" in error_msg.lower():
                # This error occurs with certain torch/sentence-transformers version combinations
                # Try alternative loading approach
                try:
                    # Clear any cached model state
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Try with a different model that's more stable
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs
                    )
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Failed to load embedding model. This may be a version conflict.\n"
                        f"Try running: pip install --upgrade sentence-transformers torch transformers\n"
                        f"Original error: {error_msg}\n"
                        f"Fallback error: {str(fallback_error)}"
                    )
            else:
                raise
        
    def load_and_process_documents(self, urls: List[str], progress_callback=None):
        """Load, split, and index documents"""
        if progress_callback:
            progress_callback("Loading documents from URLs...")
        
        docs = []
        for i, url in enumerate(urls):
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
                if progress_callback:
                    progress_callback(f"Loaded {i+1}/{len(urls)} documents")
            except Exception as e:
                st.warning(f"Failed to load {url}: {str(e)}")
        
        if progress_callback:
            progress_callback("Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        
        return doc_splits
    
    def create_vector_store(self):
        """Initialize Astra DB vector store"""
        self.vector_store = Cassandra(
            embedding=self.embeddings,
            table_name="intelligent_knowledge_base",
            session=None,
            keyspace=None
        )
        return self.vector_store
    
    def add_documents(self, documents: List[Document], progress_callback=None):
        """Add documents to vector store"""
        if progress_callback:
            progress_callback("Indexing documents in Astra DB...")
        
        self.vector_store.add_documents(documents)
        
        if progress_callback:
            progress_callback(f"Successfully indexed {len(documents)} document chunks")

class IntelligentRouter:
    """LLM-powered query router"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.llm = None
        self.question_router = None
        self.generation_chain = None
        
    def initialize(self):
        """Set up LLM and routing chain"""
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        structured_llm = self.llm.with_structured_output(RouteQuery)
        
        system_prompt = """You are an expert at routing user questions to the most relevant data source.

The vectorstore contains specialized documents about:
- AI Agents and their architectures
- Prompt Engineering techniques and best practices
- Adversarial attacks on Large Language Models
- Machine Learning security concepts

Route to 'vectorstore' for questions about these topics.
Route to 'wiki_search' for general knowledge, current events, people, places, or topics outside the vectorstore domain.

Be precise in your routing decisions."""

        route_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        self.question_router = route_prompt | structured_llm
        
        # Set up generation chain for synthesizing answers
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant specialized in providing accurate, informative answers.
            
Use the following retrieved context to answer the user's question. 
If the context doesn't contain relevant information, say so and provide general guidance.
Be concise but comprehensive. Use bullet points for clarity when appropriate.

Context:
{context}"""),
            ("human", "{question}")
        ])
        
        self.generation_chain = generation_prompt | self.llm | StrOutputParser()
    
    def route(self, question: str) -> str:
        """Route question to appropriate data source"""
        result = self.question_router.invoke({"question": question})
        return result.datasource
    
    def generate_response(self, question: str, documents: List[Document]) -> str:
        """Generate a coherent response from retrieved documents"""
        # Format documents into context string
        if isinstance(documents, list):
            context = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(documents[:5])
            ])
        else:
            context = documents.page_content if hasattr(documents, 'page_content') else str(documents)
        
        response = self.generation_chain.invoke({
            "context": context,
            "question": question
        })
        return response

class AdaptiveRAGWorkflow:
    """LangGraph-based adaptive retrieval workflow"""
    
    def __init__(self, vector_store, router: IntelligentRouter):
        self.vector_store = vector_store
        self.router = router
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        self.wiki = self._setup_wikipedia()
        self.workflow = None
        self.app = None
        
    def _setup_wikipedia(self):
        """Initialize Wikipedia search tool"""
        api_wrapper = WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=1000
        )
        return WikipediaQueryRun(api_wrapper=api_wrapper)
    
    def retrieve(self, state: Dict) -> Dict:
        """Retrieve from vector store"""
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question, "route": "vectorstore"}
    
    def wiki_search(self, state: Dict) -> Dict:
        """Search Wikipedia"""
        question = state["question"]
        try:
            docs = self.wiki.invoke({"query": question})
            wiki_results = Document(page_content=docs)
        except Exception as e:
            wiki_results = Document(page_content=f"Wikipedia search returned no results for this query. Error: {str(e)}")
        return {"documents": [wiki_results], "question": question, "route": "wikipedia"}
    
    def generate(self, state: Dict) -> Dict:
        """Generate response from retrieved documents"""
        question = state["question"]
        documents = state["documents"]
        
        # Use the router's generation chain to create a response
        generation = self.router.generate_response(question, documents)
        
        return {
            "question": question,
            "documents": documents,
            "generation": generation,
            "route": state.get("route", "unknown")
        }
    
    def route_question(self, state: Dict) -> str:
        """Route based on question type"""
        question = state["question"]
        source = self.router.route(question)
        
        if source == "wiki_search":
            return "wiki_search"
        else:
            return "vectorstore"
    
    def build_graph(self):
        """Construct LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("wiki_search", self.wiki_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        
        # Add conditional edges from START
        workflow.add_conditional_edges(
            START,
            self.route_question,
            {
                "wiki_search": "wiki_search",
                "vectorstore": "retrieve",
            },
        )
        
        # Both retrieval paths lead to generation
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("wiki_search", "generate")
        
        # Generation leads to END
        workflow.add_edge("generate", END)
        
        self.app = workflow.compile()
        
    def execute(self, question: str) -> Dict[str, Any]:
        """Execute workflow and return results"""
        inputs = {"question": question}
        
        result = {
            "route": None,
            "documents": [],
            "generation": "",
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            for output in self.app.stream(inputs):
                for key, value in output.items():
                    if key == "generate":
                        result["generation"] = value.get("generation", "")
                        result["route"] = value.get("route", "unknown")
                        result["documents"] = value.get("documents", [])
                    elif key in ["retrieve", "wiki_search"]:
                        result["route"] = value.get("route", key)
                        result["documents"] = value.get("documents", [])
        except Exception as e:
            result["generation"] = f"Error executing query: {str(e)}"
            result["route"] = "error"
        
        result["execution_time"] = time.time() - start_time
        
        return result

# ==================== Streamlit UI ====================

def render_header():
    """Render stunning animated header"""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title floating">🧠 IMSKOS</div>
        <div class="hero-subtitle">Intelligent Multi-Source Knowledge Orchestration System</div>
        <div class="hero-badge">⚡ Powered by LangGraph • Astra DB • Groq LLM</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature pills
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; margin-bottom: 1rem;">
        <span class="feature-pill">🔄 Adaptive Routing</span>
        <span class="feature-pill">🗄️ Vector Storage</span>
        <span class="feature-pill">📖 Wikipedia Search</span>
        <span class="feature-pill">⚡ Real-time Processing</span>
        <span class="feature-pill">🎯 Semantic Search</span>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render stylish sidebar with configuration and info"""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧠</div>
            <div style="font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 700; 
                        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                IMSKOS
            </div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0.25rem;">
                v2.0 • Knowledge Orchestrator
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System Status
        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
                ⚙️ System Status
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicators
        if st.session_state.get('initialized'):
            st.success("✅ System Online")
        else:
            st.warning("⏳ Initializing...")
        
        if st.session_state.get('db_connected'):
            st.success("🗄️ Astra DB Connected")
        else:
            st.info("🗄️ DB: Connect on Index")
            
        if st.session_state.get('documents_indexed'):
            st.success(f"📚 {st.session_state.get('num_documents', 0)} docs indexed")
        else:
            st.info("📚 No documents indexed")
        
        st.markdown("---")
        
        # Tech Stack
        st.markdown("""
        <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
            🛠️ Technology Stack
        </div>
        """, unsafe_allow_html=True)
        
        tech_items = [
            ("🔗", "LangGraph", "Workflow Orchestration"),
            ("🗄️", "Astra DB", "Vector Storage"),
            ("⚡", "Groq LLM", "Fast Inference"),
            ("🤗", "HuggingFace", "Embeddings"),
            ("📖", "Wikipedia", "External Knowledge"),
        ]
        
        for emoji, name, desc in tech_items:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; 
                        padding: 0.5rem; margin: 0.25rem 0;
                        background: rgba(255,255,255,0.03); border-radius: 8px;">
                <span style="font-size: 1.2rem;">{emoji}</span>
                <div>
                    <div style="color: white; font-weight: 500; font-size: 0.9rem;">{name}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Capabilities
        st.markdown("""
        <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
            🎯 Capabilities
        </div>
        """, unsafe_allow_html=True)
        
        capabilities = [
            "Adaptive Query Routing",
            "Semantic Vector Search",
            "Multi-Source Fusion",
            "Real-time Processing",
            "Intelligent Responses"
        ]
        
        for cap in capabilities:
            st.markdown(f"""
            <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem; 
                        padding: 0.25rem 0; display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #10b981;">✓</span> {cap}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Reset Button
        reset_clicked = st.button("🔄 Reset System", use_container_width=True)
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-top: 1rem;">
            <div style="color: rgba(255,255,255,0.4); font-size: 0.75rem;">
                Built with ❤️ by 
                <a href="https://github.com/KUNALSHAWW" target="_blank" 
                   style="color: #a78bfa; text-decoration: none;">Kunal Shaw</a>
            </div>
            <div style="margin-top: 0.5rem;">
                <a href="https://github.com/KUNALSHAWW/IMSKOS" target="_blank"
                   style="color: rgba(255,255,255,0.5); font-size: 0.8rem; text-decoration: none;">
                    ⭐ Star on GitHub
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return reset_clicked

def initialize_system():
    """Initialize all system components with stylish loading (lazy DB connection)"""
    if 'initialized' not in st.session_state:
        # Animated loading container
        loading_container = st.empty()
        progress_bar = st.progress(0)
        
        loading_container.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🚀</div>
            <div style="color: white; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
                Initializing IMSKOS
            </div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
                {random.choice(LOADING_QUOTES)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Step 1: Load configuration (20%)
            progress_bar.progress(10)
            time.sleep(0.3)
            config = Config.load_env_variables()
            progress_bar.progress(20)
            
            # Step 2: Initialize Knowledge Base Manager (without DB connection - lazy)
            kb_manager = KnowledgeBaseManager(
                config["ASTRA_DB_APPLICATION_TOKEN"],
                config["ASTRA_DB_ID"]
            )
            progress_bar.progress(30)
            # NOTE: Skipping cassio.init() here - will connect when indexing documents
            # This allows the app to start even if Astra DB is hibernated
            progress_bar.progress(40)
            kb_manager.setup_embeddings()
            progress_bar.progress(50)
            
            # Step 3: Initialize Router (80%)
            router = IntelligentRouter(config["GROQ_API_KEY"])
            progress_bar.progress(60)
            router.initialize()
            progress_bar.progress(80)
            
            # Step 4: Store in session state (100%)
            st.session_state.kb_manager = kb_manager
            st.session_state.router = router
            st.session_state.config = config
            st.session_state.initialized = True
            st.session_state.documents_indexed = False
            st.session_state.db_connected = False
            progress_bar.progress(100)
            
            # Clear loading and show success
            loading_container.empty()
            progress_bar.empty()
            
            st.markdown("""
            <div class="success-box">
                <span style="font-size: 1.2rem;">✅</span>
                <strong>System Online!</strong> IMSKOS is ready. Connect to Astra DB when indexing documents.
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            loading_container.empty()
            progress_bar.empty()
            st.markdown(f"""
            <div class="warning-box">
                <span style="font-size: 1.2rem;">❌</span>
                <strong>Initialization Failed</strong><br>
                <code>{str(e)}</code>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

def render_indexing_tab():
    """Render beautiful document indexing interface"""
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            📚 Knowledge Base Indexing
        </div>
        <div class="glass-card-content">
            Index domain-specific documents to create your proprietary knowledge base. 
            The system uses advanced chunking strategies and distributed vector storage 
            for optimal retrieval performance.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works section
    with st.expander("🔍 How Document Indexing Works", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📥</div>
                <div style="color: #a78bfa; font-weight: 600;">1. Load</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                    Fetch content from URLs using web scraping
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">✂️</div>
                <div style="color: #a78bfa; font-weight: 600;">2. Chunk</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                    Split into semantic chunks with overlap
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🗄️</div>
                <div style="color: #a78bfa; font-weight: 600;">3. Index</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                    Generate embeddings & store in Astra DB
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # URL input section
    st.markdown("""
    <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.5rem;">
        🔗 Document Sources
    </div>
    """, unsafe_allow_html=True)
    
    default_urls = Config.get_default_urls()
    
    urls_text = st.text_area(
        "Enter URLs to index (one per line):",
        value="\n".join(default_urls),
        height=150,
        label_visibility="collapsed",
        placeholder="https://example.com/document1\nhttps://example.com/document2"
    )
    
    urls = [url.strip() for url in urls_text.split("\n") if url.strip()]
    
    # Metrics display
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.5rem 0;">
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(urls)}</div>
            <div class="metric-label">URLs Configured</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">500</div>
            <div class="metric-label">Chunk Size (tokens)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">384</div>
            <div class="metric-label">Vector Dimensions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # HOTFIX: Show mock mode banner if applicable
    is_mock = show_mock_mode_banner()
    
    # HOTFIX: Show disabled banner if ENABLE_INDEX_BACKGROUND is false
    if not ENABLE_INDEX_BACKGROUND:
        st.error(
            "🚫 **Indexing Temporarily Disabled**\n\n"
            "Document indexing is currently disabled via `ENABLE_INDEX_BACKGROUND=false`. "
            "This is a temporary safety measure. Contact your administrator to re-enable."
        )
        return
    
    # Index button
    if st.button("🚀 Index Documents", type="primary", use_container_width=True):
        if not urls:
            st.markdown("""
            <div class="warning-box">
                ⚠️ Please provide at least one URL to index
            </div>
            """, unsafe_allow_html=True)
            return
        
        progress_container = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_status(message, progress_pct):
            status_text.markdown(f"""
            <div class="info-box">
                <span style="margin-right: 0.5rem;">⏳</span> {message}
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(progress_pct)
        
        # =================================================================
        # HOTFIX: Wrap all indexing operations in try/except
        # Use safe_* functions that don't raise and keep UI responsive
        # =================================================================
        try:
            kb_manager = st.session_state.kb_manager
            
            # Step 1: Connect to Astra DB using SAFE function
            if not st.session_state.get('db_connected', False) and not MOCK_MODE:
                update_status("Connecting to Astra DB (this may take a moment if DB was hibernated)...", 5)
                
                # HOTFIX: Use safe wrapper that won't crash
                init_result = safe_initialize_cassandra(kb_manager)
                
                if init_result.get("success"):
                    st.session_state.db_connected = True
                    if init_result.get("mock"):
                        update_status("Mock mode: DB connection skipped", 10)
                else:
                    # Show error but don't crash
                    progress_bar.empty()
                    status_text.empty()
                    st.markdown(f"""
                    <div class="warning-box">
                        <strong>⚠️ Astra DB Connection Failed</strong><br><br>
                        <div style="margin-bottom: 1rem;">{init_result.get('error', 'Unknown error')}</div>
                        <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px; font-size: 0.9rem;">
                            <strong>💡 Solutions:</strong>
                            <ol style="margin: 0.5rem 0; padding-left: 1.5rem;">
                                <li>Go to <a href="https://astra.datastax.com" target="_blank" style="color: #a78bfa;">astra.datastax.com</a></li>
                                <li>Check if your database is <strong>Active</strong> (not Hibernated)</li>
                                <li>If hibernated, click <strong>"Resume"</strong> and wait 1-2 minutes</li>
                                <li>Or set <code>MOCK_MODE=true</code> to test without DB</li>
                            </ol>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Show traceback in expander
                    if init_result.get("traceback"):
                        with st.expander("🔍 Error Details (Technical)"):
                            st.code(init_result["traceback"], language="python")
                    return
            elif MOCK_MODE:
                update_status("Mock mode: Skipping DB connection", 10)
                st.session_state.db_connected = True
            
            # Step 2: Load documents using SAFE function with timeout
            update_status("Loading documents from URLs...", 15)
            
            # HOTFIX: Use safe document loading that won't block indefinitely
            load_result = safe_load_and_process_documents(kb_manager, urls, lambda msg: update_status(msg, 30))
            
            if not load_result.get("success"):
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Failed to load documents: {load_result.get('error', 'Unknown error')}")
                if load_result.get("traceback"):
                    with st.expander("🔍 Error Details"):
                        st.code(load_result["traceback"], language="python")
                return
            
            doc_splits = load_result.get("doc_splits", [])
            is_mock_data = load_result.get("mock", False)
            
            update_status(f"{'[MOCK] ' if is_mock_data else ''}Loaded and chunked into {len(doc_splits)} pieces", 50)
            
            # Step 3: Create vector store (only if not mock mode)
            if not MOCK_MODE:
                if not kb_manager.vector_store:
                    update_status("Initializing Astra DB vector store...", 60)
                    try:
                        kb_manager.create_vector_store()
                    except Exception as vs_error:
                        logging.exception("HOTFIX: Vector store creation failed")
                        st.error(f"❌ Vector store initialization failed: {str(vs_error)}")
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc(), language="python")
                        return
                
                # Step 4: Add documents with timeout protection
                update_status("Generating embeddings and indexing...", 70)
                try:
                    # Use threaded execution with timeout for embedding generation
                    add_result = run_indexing_with_timeout(
                        kb_manager.add_documents,
                        doc_splits,
                        lambda msg: None,
                        timeout_seconds=120  # 2 minute timeout for adding docs
                    )
                    
                    if not add_result.get("success"):
                        if add_result.get("timeout"):
                            st.warning(
                                "⏱️ Embedding generation timed out. Try with fewer documents or use background mode."
                            )
                        else:
                            st.error(f"❌ Failed to add documents: {add_result.get('error')}")
                        if add_result.get("traceback"):
                            with st.expander("🔍 Error Details"):
                                st.code(add_result["traceback"], language="python")
                        return
                        
                except Exception as add_error:
                    logging.exception("HOTFIX: Document addition failed")
                    st.error(f"❌ Failed to index documents: {str(add_error)}")
                    return
            else:
                # Mock mode: simulate indexing
                update_status("[MOCK] Simulating embedding generation...", 70)
                time.sleep(0.5)  # Brief delay to simulate work
            
            update_status("Finalizing index...", 90)
            
            # Update session state
            st.session_state.documents_indexed = True
            st.session_state.num_documents = len(doc_splits)
            st.session_state.index_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            progress_bar.progress(100)
            status_text.empty()
            
            # Success message
            mock_prefix = "[MOCK MODE] " if MOCK_MODE or is_mock_data else ""
            st.markdown(f"""
            <div class="success-box">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">🎉</span>
                    <strong>{mock_prefix}Indexing Complete!</strong>
                </div>
                <div style="color: rgba(255,255,255,0.8);">
                    Successfully processed <strong>{len(doc_splits)}</strong> document chunks 
                    {'(mock data)' if is_mock_data else 'and stored them in Astra DB vector database'}.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Results metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Chunks", len(doc_splits))
            with col2:
                st.metric("📄 URLs Processed", len(urls))
            with col3:
                st.metric("🔢 Dimensions", 384)
            with col4:
                st.metric("⚡ Status", "Ready" if not MOCK_MODE else "Mock")
            
        except Exception as e:
            # HOTFIX: Catch-all exception handler - UI must never crash
            logging.exception("HOTFIX: Unhandled exception in indexing flow")
            progress_bar.empty()
            status_text.empty()
            st.markdown(f"""
            <div class="warning-box">
                <strong>❌ Indexing Failed</strong><br>
                <code>{str(e)}</code>
            </div>
            """, unsafe_allow_html=True)
            # Show full traceback for debugging
            with st.expander("🔍 Full Error Details"):
                st.code(traceback.format_exc(), language="python")

def render_query_tab():
    """Render stunning intelligent query interface"""
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            🔍 Intelligent Query Interface
        </div>
        <div class="glass-card-content">
            Ask anything! The system automatically routes your query to the optimal data source 
            using LLM-powered intelligent routing.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.get('documents_indexed', False):
        st.markdown("""
        <div class="warning-box">
            <span style="font-size: 1.2rem;">📚</span>
            <strong>Knowledge Base Required</strong><br>
            Please index documents first in the <strong>Knowledge Base Indexing</strong> tab before querying.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Routing explanation
    with st.expander("🎯 How Intelligent Routing Works", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="glass-card" style="border-color: rgba(59, 130, 246, 0.3);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">🗄️</span>
                    <span style="color: #93c5fd; font-weight: 600;">Vector Store</span>
                </div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                    <strong>Best for:</strong>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>AI Agents & Architectures</li>
                        <li>Prompt Engineering</li>
                        <li>LLM Security & Attacks</li>
                        <li>Your indexed content</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="glass-card" style="border-color: rgba(245, 158, 11, 0.3);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">📖</span>
                    <span style="color: #fcd34d; font-weight: 600;">Wikipedia</span>
                </div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                    <strong>Best for:</strong>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>General Knowledge</li>
                        <li>People & Places</li>
                        <li>Current Events</li>
                        <li>Broad Topics</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Example queries
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
        💡 Try These Example Queries
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <div style="color: #93c5fd; font-size: 0.85rem; margin-bottom: 0.5rem;">🗄️ Vector Store Queries</div>
            <code style="display: block; margin: 0.25rem 0;">What are the types of agent memory?</code>
            <code style="display: block; margin: 0.25rem 0;">Explain chain of thought prompting</code>
            <code style="display: block; margin: 0.25rem 0;">How do adversarial attacks work?</code>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <div style="color: #fcd34d; font-size: 0.85rem; margin-bottom: 0.5rem;">📖 Wikipedia Queries</div>
            <code style="display: block; margin: 0.25rem 0;">Who is Elon Musk?</code>
            <code style="display: block; margin: 0.25rem 0;">What is quantum computing?</code>
            <code style="display: block; margin: 0.25rem 0;">Tell me about the Avengers</code>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Query input
    st.markdown("""
    <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.5rem;">
        🤔 Enter Your Question
    </div>
    """, unsafe_allow_html=True)
    
    query = st.text_input(
        "Your question:",
        placeholder="e.g., What is an AI agent and how does it work?",
        key="query_input",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button("🚀 Execute Query", type="primary", use_container_width=True)
    with col2:
        advanced_mode = st.checkbox("🔬 Advanced", help="Show additional metadata and debug info")
    
    if search_button and query:
        # Animated processing indicator
        with st.spinner(""):
            processing_placeholder = st.empty()
            processing_placeholder.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 1.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;" class="floating">🤖</div>
                <div style="color: white; font-weight: 600;">{random.choice(LOADING_QUOTES)}</div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Build workflow if not exists
                if 'rag_workflow' not in st.session_state:
                    kb_manager = st.session_state.kb_manager
                    router = st.session_state.router
                    
                    rag_workflow = AdaptiveRAGWorkflow(
                        kb_manager.vector_store,
                        router
                    )
                    rag_workflow.build_graph()
                    st.session_state.rag_workflow = rag_workflow
                
                # Execute query
                workflow = st.session_state.rag_workflow
                result = workflow.execute(query)
                
                processing_placeholder.empty()
                
                # Display routing badge
                route = result["route"]
                route_class = "route-vector" if route == "vectorstore" else "route-wiki"
                route_emoji = "🗄️" if route == "vectorstore" else "📖"
                route_name = "Vector Store" if route == "vectorstore" else "Wikipedia"
                
                # Results header with metrics
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="route-badge {route_class}">
                        {route_emoji} Routed to {route_name}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.metric("⚡ Processing Time", f"{result['execution_time']:.2f}s")
                with col3:
                    num_docs = len(result['documents']) if isinstance(result['documents'], list) else 1
                    st.metric("📄 Sources Found", num_docs)
                
                # AI Response
                st.markdown(f"""
                <div class="response-container">
                    <div class="response-header">
                        <span>🤖</span> AI-Generated Response
                    </div>
                    <div class="response-text">
                        {result['generation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Source Documents
                st.markdown("""
                <div style="color: #a78bfa; font-weight: 600; margin: 1.5rem 0 0.75rem 0;">
                    📄 Source Documents
                </div>
                """, unsafe_allow_html=True)
                
                documents = result['documents']
                if isinstance(documents, list) and documents:
                    for i, doc in enumerate(documents[:5], 1):
                        with st.expander(f"📌 Source {i}", expanded=False):
                            if hasattr(doc, 'page_content'):
                                st.markdown(f"""
                                <div class="source-card">
                                    {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(str(doc)[:500])
                            
                            if advanced_mode and hasattr(doc, 'metadata') and doc.metadata:
                                st.markdown("**Metadata:**")
                                st.json(doc.metadata)
                elif hasattr(documents, 'page_content'):
                    with st.expander("📌 Source Document", expanded=False):
                        st.markdown(documents.page_content)
                
                # Store in query history
                if 'query_history' not in st.session_state:
                    st.session_state.query_history = []
                
                st.session_state.query_history.append({
                    "query": query,
                    "route": route_name,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "execution_time": result['execution_time'],
                    "response_preview": result['generation'][:100] + "..." if len(result['generation']) > 100 else result['generation']
                })
                
            except Exception as e:
                processing_placeholder.empty()
                st.markdown(f"""
                <div class="warning-box">
                    <strong>❌ Query Failed</strong><br>
                    <code>{str(e)}</code>
                </div>
                """, unsafe_allow_html=True)
                if st.checkbox("Show error details"):
                    st.code(traceback.format_exc())

def render_analytics_tab():
    """Render beautiful system analytics and monitoring dashboard"""
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            📈 System Analytics & Monitoring
        </div>
        <div class="glass-card-content">
            Real-time insights into query patterns, routing decisions, and system performance.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'query_history' not in st.session_state or not st.session_state.query_history:
        st.markdown("""
        <div class="info-box">
            <span style="font-size: 1.2rem;">📊</span>
            <strong>No Analytics Available</strong><br>
            Execute some queries to see analytics and insights here!
        </div>
        """, unsafe_allow_html=True)
        
        # Show system info anyway
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
            💾 Knowledge Base Status
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            indexed = st.session_state.get('documents_indexed', False)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'✅' if indexed else '❌'}</div>
                <div class="metric-label">Index Status</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.get('num_documents', 0)}</div>
                <div class="metric-label">Document Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">384</div>
                <div class="metric-label">Vector Dims</div>
            </div>
            """, unsafe_allow_html=True)
        return
    
    history = st.session_state.query_history
    
    # Main metrics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
        📊 Query Statistics
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_queries = len(history)
    vector_count = sum(1 for h in history if h['route'] == 'Vector Store')
    wiki_count = sum(1 for h in history if h['route'] == 'Wikipedia')
    avg_time = sum(h['execution_time'] for h in history) / len(history)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_queries}</div>
            <div class="metric-label">Total Queries</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{vector_count}</div>
            <div class="metric-label">Vector Store</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{wiki_count}</div>
            <div class="metric-label">Wikipedia</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_time:.2f}s</div>
            <div class="metric-label">Avg Response</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Routing distribution visualization
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
            🎯 Routing Distribution
        </div>
        """, unsafe_allow_html=True)
        
        if total_queries > 0:
            vector_pct = (vector_count / total_queries) * 100
            wiki_pct = (wiki_count / total_queries) * 100
            
            st.markdown(f"""
            <div class="glass-card" style="padding: 1rem;">
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: #93c5fd;">🗄️ Vector Store</span>
                        <span style="color: #93c5fd;">{vector_pct:.1f}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%); 
                                    height: 100%; width: {vector_pct}%; border-radius: 10px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: #fcd34d;">📖 Wikipedia</span>
                        <span style="color: #fcd34d;">{wiki_pct:.1f}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%); 
                                    height: 100%; width: {wiki_pct}%; border-radius: 10px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
            ⚡ Performance Metrics
        </div>
        """, unsafe_allow_html=True)
        
        fastest = min(h['execution_time'] for h in history)
        slowest = max(h['execution_time'] for h in history)
        
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem;">
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; 
                        border-bottom: 1px solid rgba(255,255,255,0.1);">
                <span style="color: rgba(255,255,255,0.7);">Fastest Query</span>
                <span style="color: #10b981; font-family: 'JetBrains Mono', monospace;">{fastest:.2f}s</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; 
                        border-bottom: 1px solid rgba(255,255,255,0.1);">
                <span style="color: rgba(255,255,255,0.7);">Slowest Query</span>
                <span style="color: #f59e0b; font-family: 'JetBrains Mono', monospace;">{slowest:.2f}s</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                <span style="color: rgba(255,255,255,0.7);">Average Time</span>
                <span style="color: #a78bfa; font-family: 'JetBrains Mono', monospace;">{avg_time:.2f}s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Query history table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
        📜 Recent Query History
    </div>
    """, unsafe_allow_html=True)
    
    import pandas as pd
    df = pd.DataFrame(history)
    df = df.rename(columns={
        'query': '🔍 Query',
        'route': '🎯 Route', 
        'timestamp': '🕐 Time',
        'execution_time': '⚡ Duration',
        'response_preview': '💬 Response Preview'
    })
    
    # Style the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "⚡ Duration": st.column_config.NumberColumn(format="%.2f s"),
        }
    )
    
    # Knowledge base status
    if st.session_state.get('documents_indexed'):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.75rem;">
            💾 Knowledge Base Status
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Document Chunks", st.session_state.get('num_documents', 0))
        with col2:
            st.metric("🕐 Last Indexed", st.session_state.get('index_timestamp', 'N/A'))
        with col3:
            st.metric("⚡ Status", "Online")

def main():
    """Main application entry point"""
    render_header()
    
    # Sidebar
    reset_clicked = render_sidebar()
    if reset_clicked:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Initialize system
    initialize_system()
    
    # Main tabs with custom styling
    tabs = st.tabs([
        "📚 Knowledge Base",
        "🔍 Query Engine", 
        "📈 Analytics"
    ])
    
    with tabs[0]:
        render_indexing_tab()
    
    with tabs[1]:
        render_query_tab()
    
    with tabs[2]:
        render_analytics_tab()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
        <div style="font-weight: 600; color: rgba(255,255,255,0.7); margin-bottom: 0.25rem;">
            IMSKOS v2.0
        </div>
        <div style="font-size: 0.85rem; color: rgba(255,255,255,0.5); margin-bottom: 1rem;">
            Intelligent Multi-Source Knowledge Orchestration System
        </div>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span class="feature-pill" style="font-size: 0.75rem;">⚡ LangGraph</span>
            <span class="feature-pill" style="font-size: 0.75rem;">🗄️ Astra DB</span>
            <span class="feature-pill" style="font-size: 0.75rem;">🚀 Groq LLM</span>
            <span class="feature-pill" style="font-size: 0.75rem;">🤗 HuggingFace</span>
        </div>
        <div style="margin-top: 1.5rem; font-size: 0.8rem;">
            Made with ❤️ by <a href="https://github.com/KUNALSHAWW">Kunal Shaw</a> | 
            <a href="https://github.com/KUNALSHAWW/IMSKOS">GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
