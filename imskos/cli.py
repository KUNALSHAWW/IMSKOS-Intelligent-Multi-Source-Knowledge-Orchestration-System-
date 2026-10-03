"""Command line: ``python -m imskos ingest|ask|status|serve|eval``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings


def _engine(a):
    from .engine import Engine

    kw = {}
    if getattr(a, "provider", None):
        kw["provider"] = a.provider
    if getattr(a, "model", None):
        kw["model"] = a.model
    if getattr(a, "store_dir", None):
        kw["store_dir"] = a.store_dir
    return Engine(Settings.from_env(**kw))


def _save(out: Path, name: str, data: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {out / name}")


def _eval(a) -> None:
    from .embed import CrossEncoderReranker, make_embedder
    from .eval import e2e, injection, retrieval
    from .eval.corpus import corpus_fingerprint, load_corpus
    from .llm import make_client

    docs = load_corpus(a.corpus_cache)
    out = Path(a.out)
    embedder = make_embedder(a.embedding_model)
    fp = corpus_fingerprint(docs)
    missing = retrieval.check_gold(docs, retrieval.load_questions())
    if missing:
        sys.exit(f"gold phrases not found in the corpus for: {missing} (corpus changed?)")

    if a.suite in ("retrieval", "all"):
        rr = CrossEncoderReranker(a.rerank_model) if a.rerank_model else None
        res = retrieval.run_retrieval(docs, embedder, rr)
        res["sweep"] = retrieval.run_chunk_sweep(docs, embedder)
        res["corpus_sha256"] = fp
        _save(out, "retrieval.json", res)
        for m, v in res["modes"].items():
            print(f"{m:15s} hit@1 {v['hit@1']:.2f} hit@5 {v['hit@5']:.2f} mrr {v['mrr@10']:.3f} ndcg {v['ndcg@10']:.3f}")

    if a.suite in ("e2e", "injection", "all"):
        base = Settings.from_env(provider=a.provider, model=a.model or "", store="local", router=a.router)
        llm = make_client(base)
        print("LLM:", llm.name)
        if a.suite in ("e2e", "all"):
            sets = ["answerable", "unanswerable"]
            out_cfg: dict = {"llm": llm.name, "embedder": embedder.name, "corpus_sha256": fp, "router": a.router}
            if a.configs == "both":
                out_cfg["baseline"] = e2e.run_config("baseline", e2e.build_engine(docs, llm, embedder, base, **e2e.BASELINE), sets)
            out_cfg["full"] = e2e.run_config("full", e2e.build_engine(docs, llm, embedder, base), sets + ["web"])
            name = "e2e.json" if a.router == "evidence" else f"e2e_{a.router}.json"
            _save(out, name, out_cfg)
            print(json.dumps({k: v["summary"] for k, v in out_cfg.items() if isinstance(v, dict) and "summary" in v}, indent=2))
        if a.suite in ("injection", "all"):
            res = injection.run_injection(docs, llm, embedder, base)
            res.update(llm=llm.name, corpus_sha256=fp)
            _save(out, "injection.json", res)
            print(json.dumps(res["summary"], indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="imskos", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--provider", choices=["groq", "openai", "ollama", "ollama_cloud"])
        sp.add_argument("--model")
        sp.add_argument("--store-dir", dest="store_dir")

    sp = sub.add_parser("ingest", help="add URLs or files to the knowledge base")
    sp.add_argument("sources", nargs="+")
    common(sp)
    sp = sub.add_parser("ask", help="ask one question")
    sp.add_argument("question")
    sp.add_argument("--trace", action="store_true")
    common(sp)
    sp = sub.add_parser("status", help="show what is indexed")
    common(sp)
    sp = sub.add_parser("serve", help="start the REST API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp = sub.add_parser("eval", help="run benchmark suites")
    sp.add_argument("suite", choices=["retrieval", "e2e", "injection", "all"])
    sp.add_argument("--provider", default="ollama", choices=["groq", "openai", "ollama", "ollama_cloud"])
    sp.add_argument("--model", default="")
    sp.add_argument("--embedding-model", default=Settings().embedding_model)
    sp.add_argument("--router", default="evidence", choices=["evidence", "titles", "none"])
    sp.add_argument("--configs", default="both", choices=["both", "full"])
    sp.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    sp.add_argument("--corpus-cache", default="corpus_cache")
    sp.add_argument("--out", default="benchmarks/results")
    a = p.parse_args(argv)

    if a.cmd == "ingest":
        eng = _engine(a)
        for src in a.sources:
            r = eng.ingest_url(src) if src.startswith("http") else eng.ingest_file(src)
            print(f"{r.status:9s} {r.chunks:4d} chunks  {r.source}")
    elif a.cmd == "ask":
        eng = _engine(a)
        ans = eng.ask(a.question)
        print(ans.text)
        for c in ans.citations:
            print(f"  [{c.n}] {c.title or c.source}  support={c.support}\n      \"{c.quote[:160]}\"")
        print(f"route={ans.route} grounded={ans.grounded} support={ans.support} calls={ans.llm_calls} {ans.latency_s}s")
        if a.trace:
            for t in ans.trace:
                print("  ", t)
    elif a.cmd == "status":
        print(json.dumps(_engine(a).status(), indent=2))
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("imskos.api:app", host=a.host, port=a.port)
    elif a.cmd == "eval":
        _eval(a)
