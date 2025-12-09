"""
🤖 Intelligent Multi-Source Knowledge Orchestration System (IMSKOS)
================================================================
Advanced Agentic RAG Framework with Dynamic Routing & Distributed Vector Storage

An enterprise-grade, production-ready intelligent query routing system that leverages:
- LangGraph for stateful workflow orchestration
- DataStax Astra DB for distributed vector storage
- Groq LLM for high-performance inference
- Adaptive routing between proprietary knowledge base and Wikipedia
- Real-time semantic search with HuggingFace embeddings
"""

import streamlit as st
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set USER_AGENT to suppress warnings from web loaders
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "IMSKOS/1.0 (Intelligent Multi-Source Knowledge Orchestration System)"

# Compatibility shim for different typing.ForwardRef._evaluate signatures
# ------------------------------------------------------------
# Some Python/typing/pydantic versions expect ForwardRef._evaluate to accept
# recursive_guard as a keyword-only argument, while other versions accept it
# positionally. When a third-party library calls ForwardRef._evaluate using the
# older calling convention, it can raise:
#   TypeError: ForwardRef._evaluate() missing 1 required keyword-only argument: 'recursive_guard'
#
# This shim wraps/monkeypatches typing.ForwardRef._evaluate so it accepts both
# calling conventions. It should be safe and only applied at import time.
try:
    from typing import ForwardRef as _ForwardRef

    _orig_forwardref_evaluate = getattr(_ForwardRef, "_evaluate", None)
    if _orig_forwardref_evaluate is not None:
        def _evaluate_compat(self, globalns, localns, *args, **kwargs):
            """
            Compatibility wrapper that attempts to call the original _evaluate
            with whatever args/kwargs were passed. If a TypeError occurs (typical
            when the underlying implementation requires recursive_guard as
            keyword-only), call the original with recursive_guard provided as a
            keyword using the first positional arg if available or an empty set.
            """
            try:
                return _orig_forwardref_evaluate(self, globalns, localns, *args, **kwargs)
            except TypeError:
                # Older callers passed recursive_guard positionally; newer
                # implementations require recursive_guard as a keyword-only arg.
                recursive_guard = args[0] if args else set()
                return _orig_forwardref_evaluate(self, globalns, localns, recursive_guard=recursive_guard)

        # Monkeypatch the ForwardRef implementation with the compatibility wrapper
        _ForwardRef._evaluate = _evaluate_compat
except Exception:
    # If anything goes wrong here, do not prevent app import — let original
    # behavior surface later (so the original error will be visible).
    pass
# ------------------------------------------------------------

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

# Page Configuration
st.set_page_config(
    page_title="IMSKOS - Intelligent Knowledge Orchestration",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .route-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .route-vector {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .route-wiki {
        background-color: #fff3e0;
        color: #e65100;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Configuration & Initialization ====================

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
            st.error(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
            st.info("""
            **Setup Instructions:**
            1. **Local Development:** Create a `.env` file with your credentials
            2. **Streamlit Cloud:** Add secrets in the app settings
            
            Required variables:
            - `ASTRA_DB_APPLICATION_TOKEN` - Get from [DataStax Astra](https://astra.datastax.com)
            - `ASTRA_DB_ID` - Your Astra DB database ID
            - `GROQ_API_KEY` - Get from [Groq Console](https://console.groq.com)
            """)
            st.stop()
        
        return required_vars
    
    @staticmethod
    def get_default_urls():
        """Default knowledge base URLs"""
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

# ==================== Core System Classes ====================

class KnowledgeBaseManager:
    """Manages document ingestion and vector storage"""
    
    def __init__(self, astra_token: str, astra_db_id: str):
        self.astra_token = astra_token
        self.astra_db_id = astra_db_id
        self.embeddings = None
        self.vector_store = None
        
    def initialize_cassandra(self):
        """Initialize Cassandra connection"""
        cassio.init(token=self.astra_token, database_id=self.astra_db_id)
        
    def setup_embeddings(self):
        """Initialize HuggingFace embeddings"""
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
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
    """Render application header"""
    st.markdown('<h1 class="main-header">🧠 IMSKOS</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Intelligent Multi-Source Knowledge Orchestration System</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

def render_sidebar():
    """Render sidebar with configuration and info"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
        st.title("⚙️ System Configuration")
        
        st.markdown("### 🔧 Core Technologies")
        st.markdown("""
        - **LangGraph**: Stateful workflow orchestration
        - **Astra DB**: Distributed vector storage
        - **Groq**: High-performance LLM inference
        - **HuggingFace**: Semantic embeddings
        """)
        
        st.markdown("---")
        st.markdown("### 📊 System Capabilities")
        st.markdown("""
        ✅ Adaptive query routing  
        ✅ Real-time semantic search  
        ✅ Multi-source knowledge fusion  
        ✅ Scalable vector operations  
        ✅ Enterprise-grade architecture
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Use Cases")
        st.markdown("""
        - AI/ML Research Assistance
        - Technical Documentation Q&A
        - Competitive Intelligence
        - Knowledge Base Management
        """)
        
        return st.button("🔄 Reset System", use_container_width=True)

def initialize_system():
    """Initialize all system components"""
    if 'initialized' not in st.session_state:
        with st.spinner("🚀 Initializing Intelligent Knowledge Orchestration System..."):
            try:
                # Load configuration
                config = Config.load_env_variables()
                
                # Initialize Knowledge Base Manager
                kb_manager = KnowledgeBaseManager(
                    config["ASTRA_DB_APPLICATION_TOKEN"],
                    config["ASTRA_DB_ID"]
                )
                kb_manager.initialize_cassandra()
                kb_manager.setup_embeddings()
                
                # Initialize Router
                router = IntelligentRouter(config["GROQ_API_KEY"])
                router.initialize()
                
                # Store in session state
                st.session_state.kb_manager = kb_manager
                st.session_state.router = router
                st.session_state.initialized = True
                st.session_state.documents_indexed = False
                
                st.success("✅ System initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                st.stop()

def render_indexing_tab():
    """Render document indexing interface"""
    st.header("📚 Knowledge Base Indexing")
    
    st.markdown("""
    <div class="info-box">
    <strong>📌 About Knowledge Base:</strong><br>
    Index domain-specific documents to create a proprietary knowledge base. 
    The system uses advanced chunking strategies and distributed vector storage 
    for optimal retrieval performance.
    </div>
    """, unsafe_allow_html=True)
    
    # URL input
    st.subheader("🔗 Document Sources")
    default_urls = Config.get_default_urls()
    
    urls_text = st.text_area(
        "Enter URLs (one per line):",
        value="\n".join(default_urls),
        height=150
    )
    
    urls = [url.strip() for url in urls_text.split("\n") if url.strip()]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 URLs Configured", len(urls))
    with col2:
        st.metric("💾 Chunk Size", "500 tokens")
    
    if st.button("🚀 Index Documents", type="primary", use_container_width=True):
        if not urls:
            st.warning("⚠️ Please provide at least one URL")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message):
            status_text.info(message)
        
        try:
            # Load and process documents
            kb_manager = st.session_state.kb_manager
            doc_splits = kb_manager.load_and_process_documents(urls, update_progress)
            progress_bar.progress(50)
            
            # Create vector store
            if not kb_manager.vector_store:
                kb_manager.create_vector_store()
            
            # Add documents
            kb_manager.add_documents(doc_splits, update_progress)
            progress_bar.progress(100)
            
            # Store in session state
            st.session_state.documents_indexed = True
            st.session_state.num_documents = len(doc_splits)
            st.session_state.index_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            st.markdown("""
            <div class="success-box">
            ✅ <strong>Indexing Complete!</strong><br>
            Documents have been successfully processed and stored in Astra DB vector database.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Chunks", len(doc_splits))
            with col2:
                st.metric("🔢 Vector Dimensions", 384)
            with col3:
                st.metric("⚡ Status", "Ready")
            
        except Exception as e:
            st.error(f"❌ Indexing failed: {str(e)}")
            progress_bar.empty()

def render_query_tab():
    """Render intelligent query interface"""
    st.header("🔍 Intelligent Query Interface")
    
    if not st.session_state.get('documents_indexed', False):
        st.warning("⚠️ Please index documents first in the 'Knowledge Base Indexing' tab")
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>🎯 How It Works:</strong><br>
    The system automatically routes your query to the optimal data source:
    <ul>
        <li><strong>Vector Store:</strong> For AI/ML, prompt engineering, and security topics</li>
        <li><strong>Wikipedia:</strong> For general knowledge and current information</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Query examples
    with st.expander("💡 Example Queries"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Vector Store Queries:**")
            st.code("What are the types of agent memory?")
            st.code("Explain chain of thought prompting")
            st.code("How do adversarial attacks work on LLMs?")
        with col2:
            st.markdown("**Wikipedia Queries:**")
            st.code("Who is Elon Musk?")
            st.code("What is quantum computing?")
            st.code("Tell me about the Avengers")
    
    # Query input
    query = st.text_input(
        "🤔 Enter your question:",
        placeholder="e.g., What is an AI agent?",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_button = st.button("🚀 Execute Query", type="primary", use_container_width=True)
    with col2:
        advanced_mode = st.checkbox("🔬 Advanced Mode")
    
    if search_button and query:
        with st.spinner("🤖 Processing your query..."):
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
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Query Results")
                
                # Routing information
                route = result["route"]
                route_class = "route-vector" if route == "vectorstore" else "route-wiki"
                route_emoji = "🗄️" if route == "vectorstore" else "📖"
                route_name = "Vector Store" if route == "vectorstore" else "Wikipedia"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f'<div class="route-indicator {route_class}">'
                        f'{route_emoji} Route: {route_name}</div>',
                        unsafe_allow_html=True
                    )
                with col2:
                    st.metric("⚡ Execution Time", f"{result['execution_time']:.2f}s")
                with col3:
                    num_docs = len(result['documents']) if isinstance(result['documents'], list) else 1
                    st.metric("📄 Documents", num_docs)
                
                # Display AI-generated response
                st.markdown("### 🤖 AI-Generated Answer")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
                            border-left: 4px solid #667eea;">
                    {result['generation']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display source documents in expandable section
                st.markdown("### 📄 Source Documents")
                
                documents = result['documents']
                if isinstance(documents, list) and documents:
                    for i, doc in enumerate(documents[:5], 1):
                        with st.expander(f"📌 Source Document {i}", expanded=False):
                            if hasattr(doc, 'page_content'):
                                st.markdown(doc.page_content)
                            else:
                                st.markdown(str(doc))
                            
                            if advanced_mode and hasattr(doc, 'metadata') and doc.metadata:
                                st.markdown("**Metadata:**")
                                st.json(doc.metadata)
                elif hasattr(documents, 'page_content'):
                    with st.expander("📌 Source Document", expanded=False):
                        st.markdown(documents.page_content)
                
                # Store query history
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
                st.error(f"❌ Query execution failed: {str(e)}")
                if st.checkbox("Show error details"):
                    st.code(traceback.format_exc())

def render_analytics_tab():
    """Render system analytics and monitoring"""
    st.header("📈 System Analytics")
    
    if 'query_history' not in st.session_state or not st.session_state.query_history:
        st.info("📊 No queries executed yet. Run some queries to see analytics!")
        return
    
    history = st.session_state.query_history
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Queries", len(history))
    with col2:
        vector_count = sum(1 for h in history if h['route'] == 'Vector Store')
        st.metric("🗄️ Vector Store", vector_count)
    with col3:
        wiki_count = sum(1 for h in history if h['route'] == 'Wikipedia')
        st.metric("📖 Wikipedia", wiki_count)
    with col4:
        avg_time = sum(h['execution_time'] for h in history) / len(history)
        st.metric("⚡ Avg Time", f"{avg_time:.2f}s")
    
    # Query history table
    st.subheader("📜 Query History")
    import pandas as pd
    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True)
    
    # System info
    if st.session_state.get('documents_indexed'):
        st.subheader("💾 Knowledge Base Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Document Chunks", st.session_state.get('num_documents', 0))
        with col2:
            st.metric("🕐 Last Indexed", st.session_state.get('index_timestamp', 'N/A'))

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
    
    # Main tabs
    tabs = st.tabs(["📚 Knowledge Base Indexing", "🔍 Intelligent Query", "📈 Analytics"])
    
    with tabs[0]:
        render_indexing_tab()
    
    with tabs[1]:
        render_query_tab()
    
    with tabs[2]:
        render_analytics_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p><strong>IMSKOS v1.0</strong> | Built with LangGraph, Astra DB, and Groq</p>
        <p>Enterprise-Grade Intelligent Knowledge Orchestration</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
