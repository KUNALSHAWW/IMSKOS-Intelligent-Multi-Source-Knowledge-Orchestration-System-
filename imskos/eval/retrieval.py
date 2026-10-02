"""Retrieval benchmark: dense vs BM25 vs hybrid (RRF) vs hybrid + cross-encoder rerank.

A chunk is relevant to a question when it contains one of the question's gold phrases. Because the
phrases are matched against chunk text, the benchmark stays valid when the chunking changes.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..embed import Embedder
from ..retrieval import Ingestor, Retriever
from ..store import LocalStore

DATA = Path(__file__).parent / "data" / "questions.json"


def load_questions() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def is_relevant(chunk_text: str, gold_any: list[str]) -> bool:
    t = chunk_text.lower()
    return any(g in t for g in gold_any)


def check_gold(docs, questions, size: int = 900, overlap: int = 150) -> list[str]:
    """Return ids of answerable questions whose gold phrase appears in no chunk of the corpus."""
    from ..loaders import chunk_text
    chunks = [c for _, _, text in docs for c in chunk_text(text, size, overlap)]
    return [q["id"] for q in questions["answerable"] if not any(is_relevant(c, q["gold_any"]) for c in chunks)]


def _metrics(first_rank: list[int | None], rel_counts: list[list[int]], ks=(1, 3, 5, 10)) -> dict:
    n = len(first_rank)
    out = {f"hit@{k}": round(sum(r is not None and r <= k for r in first_rank) / n, 3) for k in ks}
    rr = [1 / r if r else 0.0 for r in first_rank]
    out["mrr@10"] = round(float(np.mean(rr)), 3)
    nd = []
    for rels in rel_counts:                     # rels: 0/1 relevance flags for the top-10 list
        dcg = sum(x / math.log2(i + 2) for i, x in enumerate(rels))
        total_rel = sum(rels)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(total_rel, 10)))
        nd.append(dcg / idcg if idcg else 0.0)
    out["ndcg@10"] = round(float(np.mean(nd)), 3)
    rng = np.random.default_rng(0)
    boots = [float(np.mean(rng.choice(rr, len(rr)))) for _ in range(2000)]
    out["mrr_ci95"] = [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]
    return out


def run_retrieval(docs, embedder: Embedder, reranker=None, size: int = 900, overlap: int = 150,
                  modes=("dense", "bm25", "hybrid", "hybrid+rerank")) -> dict:
    questions = load_questions()["answerable"]
    store = LocalStore()
    ing = Ingestor(store, embedder, size, overlap)
    for url, title, text in docs:
        ing.ingest_text(url, title, text)
    retr = Retriever(store, embedder, "hybrid", reranker)
    result = {"chunks": store.count(), "chunk_chars": size, "overlap": overlap, "embedder": embedder.name,
              "reranker": getattr(reranker, "name", None), "questions": len(questions), "modes": {}, "per_question": {}}
    for mode in modes:
        if mode == "hybrid+rerank" and reranker is None:
            continue
        base, rerank = (mode.split("+")[0], True) if "+" in mode else (mode, False)
        first, rels, lat = [], [], []
        for q in questions:
            t0 = time.perf_counter()
            hits = retr.search(q["q"], k=10, mode=base, rerank=rerank)
            lat.append((time.perf_counter() - t0) * 1000)
            flags = [int(is_relevant(h.chunk.text, q["gold_any"])) for h in hits]
            rank = next((i + 1 for i, f in enumerate(flags) if f), None)
            first.append(rank)
            rels.append(flags + [0] * (10 - len(flags)))
            result["per_question"].setdefault(q["id"], {})[mode] = rank
        m = _metrics(first, rels)
        m["latency_ms_p50"] = round(float(np.percentile(lat, 50)), 1)
        result["modes"][mode] = m
    return result


def run_chunk_sweep(docs, embedder: Embedder, sizes=((500, 80), (900, 150), (1500, 250))) -> list[dict]:
    rows = []
    for size, ov in sizes:
        r = run_retrieval(docs, embedder, None, size, ov, modes=("hybrid",))
        rows.append({"chunk_chars": size, "overlap": ov, "chunks": r["chunks"], **r["modes"]["hybrid"]})
    return rows
