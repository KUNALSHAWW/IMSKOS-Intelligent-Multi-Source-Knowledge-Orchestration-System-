"""Vector stores.

``LocalStore`` keeps vectors in a numpy matrix and a BM25 index in memory and persists both to a
directory, so it works offline and is what the benchmark uses. ``AstraStore`` talks to DataStax
Astra DB through its Data API and supports dense search only.

Both are idempotent per source: ingesting a document whose content hash is unchanged is a no-op, and
a changed document replaces its old chunks.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from .models import Chunk, Hit
from .textutil import tokens


class Store(Protocol):
    supports_lexical: bool

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def delete_source(self, source: str) -> int: ...
    def sources(self) -> dict[str, str]: ...          # source -> doc_hash
    def titles(self) -> list[str]: ...
    def dense(self, qvec: np.ndarray, k: int) -> list[Hit]: ...
    def lexical(self, query: str, k: int) -> list[Hit]: ...
    def count(self) -> int: ...


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        if not len(self.tf):
            return s
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(t, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class LocalStore:
    supports_lexical = True

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.chunks: list[Chunk] = []
        self.vecs = np.zeros((0, 0), dtype=np.float32)
        self._bm25: BM25 | None = None
        if self.path and (self.path / "chunks.json").exists():
            self._load()

    # persistence -----------------------------------------------------------------------------
    def _load(self) -> None:
        data = json.loads((self.path / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in data]
        self.vecs = np.load(self.path / "vectors.npy")

    def save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8")
        np.save(self.path / "vectors.npy", self.vecs)

    # write path ------------------------------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self.vecs = vectors.copy() if not len(self.chunks) else np.vstack([self.vecs, vectors])
        self.chunks.extend(chunks)
        self._bm25 = None
        self.save()

    def delete_source(self, source: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vecs = self.vecs[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self._bm25 = None
            self.save()
        return removed

    # read path -------------------------------------------------------------------------------
    def sources(self) -> dict[str, str]:
        return {c.source: c.doc_hash for c in self.chunks}

    def titles(self) -> list[str]:
        return sorted({c.title for c in self.chunks if c.title})

    def count(self) -> int:
        return len(self.chunks)

    def dense(self, qvec: np.ndarray, k: int) -> list[Hit]:
        if not self.chunks:
            return []
        sims = self.vecs @ qvec
        top = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], float(sims[i]), r + 1) for r, i in enumerate(top)]

    def lexical(self, query: str, k: int) -> list[Hit]:
        if not self.chunks:
            return []
        if self._bm25 is None:
            self._bm25 = BM25([tokens(c.text) for c in self.chunks])
        sc = self._bm25.scores(tokens(query))
        top = [i for i in np.argsort(-sc)[:k] if sc[i] > 0]
        return [Hit(self.chunks[i], float(sc[i]), r + 1) for r, i in enumerate(top)]


class AstraStore:
    """DataStax Astra DB (Data API, JSON). Dense search only; needs a token and an endpoint.

    The request shapes follow the public Data API documentation. They are covered by contract tests
    against a mock transport; they have not been exercised against a live Astra database.
    """

    supports_lexical = False

    def __init__(self, endpoint: str, token: str, collection: str = "imskos", dim: int = 384,
                 keyspace: str = "default_keyspace", client: httpx.Client | None = None):
        self.base = f"{endpoint.rstrip('/')}/api/json/v1/{keyspace}"
        self.collection = collection
        self.dim = dim
        self.http = client or httpx.Client(headers={"Token": token, "Content-Type": "application/json"}, timeout=30)
        self._ensure_collection()

    def _post(self, path: str, body: dict) -> dict:
        r = self.http.post(f"{self.base}{path}", json=body)
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            raise RuntimeError(f"Astra error: {data['errors'][0].get('message', data['errors'][0])}")
        return data

    def _ensure_collection(self) -> None:
        self._post("", {"createCollection": {"name": self.collection,
                                             "options": {"vector": {"dimension": self.dim, "metric": "cosine"}}}})

    def _coll(self, cmd: dict) -> dict:
        return self._post(f"/{self.collection}", cmd)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        for i in range(0, len(chunks), 20):          # Data API limit: 20 documents per insertMany
            docs = [{"_id": c.id, "text": c.text, "source": c.source, "title": c.title, "idx": c.idx,
                     "doc_hash": c.doc_hash, "$vector": [float(x) for x in v]}
                    for c, v in zip(chunks[i:i + 20], vectors[i:i + 20])]
            self._coll({"insertMany": {"documents": docs, "options": {"ordered": False}}})

    def delete_source(self, source: str) -> int:
        n = 0
        while True:
            d = self._coll({"deleteMany": {"filter": {"source": source}}})
            n += d.get("status", {}).get("deletedCount", 0)
            if not d.get("status", {}).get("moreData"):
                return n

    def sources(self) -> dict[str, str]:
        out: dict[str, str] = {}
        cmd: dict = {"find": {"projection": {"source": 1, "doc_hash": 1}}}
        while True:
            d = self._coll(cmd)
            for doc in d["data"]["documents"]:
                out[doc["source"]] = doc["doc_hash"]
            nxt = d["data"].get("nextPageState")
            if not nxt:
                return out
            cmd = {"find": {"projection": {"source": 1, "doc_hash": 1}, "options": {"pageState": nxt}}}

    def titles(self) -> list[str]:
        titles: set[str] = set()
        cmd: dict = {"find": {"projection": {"title": 1}}}
        while True:
            d = self._coll(cmd)
            titles.update(doc.get("title", "") for doc in d["data"]["documents"])
            nxt = d["data"].get("nextPageState")
            if not nxt:
                return sorted(t for t in titles if t)
            cmd = {"find": {"projection": {"title": 1}, "options": {"pageState": nxt}}}

    def count(self) -> int:
        return int(self._coll({"countDocuments": {}})["status"]["count"])

    def dense(self, qvec: np.ndarray, k: int) -> list[Hit]:
        d = self._coll({"find": {"sort": {"$vector": [float(x) for x in qvec]},
                                 "options": {"limit": k, "includeSimilarity": True}}})
        hits = []
        for r, doc in enumerate(d["data"]["documents"]):
            c = Chunk(doc["_id"], doc["text"], doc["source"], doc.get("title", ""), doc.get("idx", 0), doc.get("doc_hash", ""))
            hits.append(Hit(c, float(doc.get("$similarity", 0.0)), r + 1))
        return hits

    def lexical(self, query: str, k: int) -> list[Hit]:
        return []
