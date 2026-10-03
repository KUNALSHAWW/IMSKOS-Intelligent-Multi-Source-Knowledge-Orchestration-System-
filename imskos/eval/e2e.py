"""End-to-end benchmark with a real LLM: a plain RAG baseline against the full agentic graph.

* baseline  retrieve -> generate. No routing, grading, rewriting, verification or Wikipedia fallback.
* full      every node enabled.

Reported per question set: answer correctness (gold answer phrase present, not abstained), correct
abstention on questions the corpus cannot answer, routing accuracy, citation quality, cost and latency.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from ..config import Settings
from ..engine import Engine
from ..retrieval import Ingestor
from ..store import LocalStore
from .retrieval import is_relevant, load_questions

BASELINE = dict(grade=False, verify=False, max_rewrites=0, max_regenerations=0, web_fallback=False, defense=False)


def build_engine(docs, llm, embedder, settings: Settings, **overrides) -> Engine:
    s = dataclasses.replace(settings, **overrides)
    store = LocalStore()
    ing = Ingestor(store, embedder, s.chunk_chars, s.chunk_overlap)
    for url, title, text in docs:
        ing.ingest_text(url, title, text)
    return Engine(s, llm=llm, embedder=embedder, store=store)


def _correct(ans, q) -> bool:
    return (not ans.abstained) and any(p in ans.text.lower() for p in q["answer_any"])


def _citation_hit(engine: Engine, ans, q) -> bool | None:
    """Does any cited passage contain the gold evidence? None when there are no citations."""
    if not ans.citations or "gold_any" not in q:
        return None
    by_id = {c.id: c for c in engine.store.chunks}
    return any(c.chunk_id in by_id and is_relevant(by_id[c.chunk_id].text, q["gold_any"]) for c in ans.citations)


def run_config(name: str, engine: Engine, sets: list[str], log=print) -> dict:
    qs = load_questions()
    records = []
    for kind in sets:
        for q in qs[kind]:
            t0 = time.perf_counter()
            ans = engine.ask(q["q"])
            rec = {"id": q["id"], "set": kind, "answer": ans.text, "route": ans.route, "abstained": ans.abstained,
                   "grounded": ans.grounded, "support": ans.support, "llm_calls": ans.llm_calls,
                   "prompt_tokens": ans.prompt_tokens, "completion_tokens": ans.completion_tokens,
                   "latency_s": ans.latency_s, "rewrites": ans.rewrites,
                   "nodes": [t["node"] for t in ans.trace]}
            if kind in ("answerable", "web"):
                rec["correct"] = _correct(ans, q)
            if kind == "answerable":
                rec["citation_hit"] = _citation_hit(engine, ans, q)
            records.append(rec)
            log(f"[{name}] {q['id']} route={ans.route} abst={ans.abstained} "
                f"{'ok' if rec.get('correct') else ''} ({time.perf_counter() - t0:.0f}s)")
    return {"config": name, "summary": summarise(records), "records": records}


def summarise(records: list[dict]) -> dict:
    def rate(rs, key):
        return round(sum(bool(r[key]) for r in rs) / len(rs), 3) if rs else None

    by = {k: [r for r in records if r["set"] == k] for k in ("answerable", "web", "unanswerable")}
    s: dict = {"n": {k: len(v) for k, v in by.items()}}
    s["answerable_correct"] = rate(by["answerable"], "correct")
    s["answerable_abstained"] = rate(by["answerable"], "abstained")
    cites = [r["citation_hit"] for r in by["answerable"] if r.get("citation_hit") is not None]
    s["citation_hit"] = round(sum(cites) / len(cites), 3) if cites else None
    s["web_correct"] = rate(by["web"], "correct")
    s["unanswerable_abstained"] = rate(by["unanswerable"], "abstained")
    s["unanswerable_hallucinated"] = round(1 - s["unanswerable_abstained"], 3) if by["unanswerable"] else None
    routed = [(r["set"] == "web") == (r["route"] == "web") for r in records if r["set"] in ("answerable", "web")]
    s["routing_accuracy"] = round(sum(routed) / len(routed), 3) if routed else None
    lat = [r["latency_s"] for r in records]
    s["latency_s_p50"] = round(float(np.percentile(lat, 50)), 1)
    s["latency_s_p95"] = round(float(np.percentile(lat, 95)), 1)
    s["llm_calls_mean"] = round(float(np.mean([r["llm_calls"] for r in records])), 2)
    s["completion_tokens_mean"] = round(float(np.mean([r["completion_tokens"] for r in records])), 1)
    return s
