"""Wires settings, store, embedder, retriever and the agentic graph into one object."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .config import Settings
from .embed import CrossEncoderReranker, Embedder, make_embedder
from .graph import AgenticRAG
from .llm import make_client
from .loaders import fetch_url, load_file
from .models import Answer
from .retrieval import Ingestor, IngestResult, Retriever
from .store import AstraStore, LocalStore, Store
from .web import wikipedia_search


class Engine:
    def __init__(self, settings: Settings | None = None, llm=None, embedder: Embedder | None = None,
                 store: Store | None = None, web_search=None, reranker=None):
        self.settings = s = settings or Settings.from_env()
        self.embedder = embedder or make_embedder(s.embedding_model)
        if store is None:
            if s.store == "astra":
                store = AstraStore(s.astra_endpoint, s.astra_token, s.astra_collection, self.embedder.dim)
            else:
                store = LocalStore(s.store_dir)
        self.store = store
        if reranker is None and s.rerank_model:
            reranker = CrossEncoderReranker(s.rerank_model)
        self.retriever = Retriever(store, self.embedder, s.retrieval_mode, reranker)
        self.ingestor = Ingestor(store, self.embedder, s.chunk_chars, s.chunk_overlap)
        self.llm = llm or make_client(s)
        self.agent = AgenticRAG(self.llm, self.retriever, s, web_search or wikipedia_search, store.titles)

    # ingestion -------------------------------------------------------------------------------
    def ingest_url(self, url: str) -> IngestResult:
        title, text = fetch_url(url)
        return self.ingestor.ingest_text(url, title, text)

    def ingest_file(self, path: str | Path) -> IngestResult:
        title, text = load_file(path)
        return self.ingestor.ingest_text(str(path), title, text)

    def ingest_text(self, source: str, title: str, text: str) -> IngestResult:
        return self.ingestor.ingest_text(source, title, text)

    # querying --------------------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        return self.agent.ask(question)

    def stream(self, question: str) -> Iterator[tuple[str, object]]:
        return self.agent.stream(question)

    def status(self) -> dict:
        return {"chunks": self.store.count(), "sources": self.store.sources(), "titles": self.store.titles(),
                "embedder": self.embedder.name, "llm": getattr(self.llm, "name", "?"),
                "mode": self.settings.retrieval_mode, "store": self.settings.store}
