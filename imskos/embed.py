"""Embedding backends. All return L2-normalised float32 matrices, so a dot product is cosine."""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .textutil import tokens


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder.

    It has no semantic knowledge. It exists so tests, CI and offline demos run without
    downloading a model, and as a lexical baseline in the benchmark.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokens(t)
            for a, b in zip(toks, toks[1:] + [""]):
                for feat, w in ((a, 1.0), (f"{a}_{b}", 0.5)):
                    h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
                    out[i, h % self.dim] += w if (h >> 32) & 1 else -w
        return _norm(out)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # heavy import, kept lazy

        self._m = SentenceTransformer(model)
        self.name = model
        self.dim = int(self._m.get_sentence_embedding_dimension())
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self._m.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        return _norm(np.asarray(v, dtype=np.float32))


class CrossEncoderReranker:
    """Optional second-stage reranker (query, passage) -> relevance score."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._m = CrossEncoder(model)
        self.name = model

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._m.predict([(query, p) for p in passages], show_progress_bar=False)]


def make_embedder(name: str) -> Embedder:
    if name.startswith("hash"):
        return HashEmbedder()
    return SentenceTransformerEmbedder(name)
