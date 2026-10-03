"""Ingestion and retrieval: dense, BM25, and hybrid (reciprocal rank fusion) with optional reranking."""
from __future__ import annotations

from dataclasses import dataclass

from .embed import Embedder
from .loaders import make_chunks, sha256
from .models import Hit
from .store import Store

MODES = ("dense", "bm25", "hybrid")


@dataclass
class IngestResult:
    source: str
    status: str          # added | updated | unchanged
    chunks: int


class Ingestor:
    def __init__(self, store: Store, embedder: Embedder, chunk_chars: int = 900, chunk_overlap: int = 150):
        self.store, self.embedder = store, embedder
        self.size, self.overlap = chunk_chars, chunk_overlap

    def ingest_text(self, source: str, title: str, text: str) -> IngestResult:
        known = self.store.sources().get(source)
        digest = sha256(text)
        if known == digest:
            return IngestResult(source, "unchanged", 0)
        status = "added"
        if known is not None:
            self.store.delete_source(source)
            status = "updated"
        chunks = make_chunks(source, title, text, self.size, self.overlap)
        if chunks:
            self.store.add(chunks, self.embedder.encode([c.text for c in chunks]))
        return IngestResult(source, status, len(chunks))


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    score: dict[str, float] = {}
    keep: dict[str, Hit] = {}
    for lst in rankings:
        for r, h in enumerate(lst):
            score[h.chunk.id] = score.get(h.chunk.id, 0.0) + 1.0 / (k + r + 1)
            keep.setdefault(h.chunk.id, h)
    order = sorted(score, key=lambda i: -score[i])
    return [Hit(keep[i].chunk, score[i], r + 1) for r, i in enumerate(order)]


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, mode: str = "hybrid", reranker=None,
                 candidates: int = 30, rrf_k: int = 60):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.store, self.embedder, self.mode = store, embedder, mode
        self.reranker, self.candidates, self.rrf_k = reranker, candidates, rrf_k

    def search(self, query: str, k: int = 5, mode: str | None = None, rerank: bool | None = None) -> list[Hit]:
        mode = mode or self.mode
        if mode != "dense" and not self.store.supports_lexical:
            mode = "dense"
        pool = max(self.candidates, k)
        lists: list[list[Hit]] = []
        if mode in ("dense", "hybrid"):
            lists.append(self.store.dense(self.embedder.encode([query])[0], pool))
        if mode in ("bm25", "hybrid"):
            lists.append(self.store.lexical(query, pool))
        hits = lists[0] if len(lists) == 1 else rrf(lists, self.rrf_k)
        use_rerank = self.reranker is not None if rerank is None else (rerank and self.reranker is not None)
        if use_rerank and hits:
            head = hits[:pool]
            scores = self.reranker.score(query, [h.chunk.text for h in head])
            head = sorted(zip(head, scores), key=lambda x: -x[1])
            hits = [Hit(h.chunk, s, r + 1) for r, (h, s) in enumerate(head)]
        return [Hit(h.chunk, h.score, r + 1) for r, h in enumerate(hits[:k])]
