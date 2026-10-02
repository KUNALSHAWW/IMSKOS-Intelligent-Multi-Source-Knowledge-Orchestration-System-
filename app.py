"""Streamlit UI for IMSKOS: ingest, ask with a live agent trace, inspect benchmarks."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from imskos import Engine, Settings

st.set_page_config(page_title="IMSKOS", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_engine(provider: str, model: str, mode: str, grade: bool, verify: bool, defense: bool, web: bool) -> Engine:
    return Engine(Settings.from_env(provider=provider, model=model, retrieval_mode=mode, grade=grade,
                                    verify=verify, defense=defense, web_fallback=web))


with st.sidebar:
    st.title("IMSKOS")
    st.caption("Agentic RAG that grades, verifies and abstains.")
    provider = st.selectbox("LLM provider", ["groq", "openai", "ollama"])
    model = st.text_input("Model (blank = default)", "")
    mode = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"])
    st.markdown("**Agent behaviour**")
    grade = st.toggle("Grade retrieved passages", True)
    verify = st.toggle("Verify grounding", True)
    defense = st.toggle("Redact injected instructions", True)
    web = st.toggle("Wikipedia fallback", True)

try:
    engine = get_engine(provider, model, mode, grade, verify, defense, web)
except Exception as e:  # missing API key, Ollama not running, model download blocked
    st.error(f"Could not start the engine: {e}")
    st.stop()

tab_ask, tab_ingest, tab_bench = st.tabs(["Ask", "Knowledge base", "Benchmarks"])

with tab_ingest:
    stat = engine.status()
    st.metric("Indexed chunks", stat["chunks"])
    st.write({"embedder": stat["embedder"], "store": stat["store"], "documents": stat["titles"]})
    urls = st.text_area("URLs to ingest (one per line)")
    up = st.file_uploader("Or upload PDF / text / HTML", type=["pdf", "txt", "md", "html"], accept_multiple_files=True)
    if st.button("Ingest", type="primary"):
        for u in [x.strip() for x in urls.splitlines() if x.strip()]:
            try:
                r = engine.ingest_url(u)
                st.success(f"{r.status}: {u} ({r.chunks} chunks)")
            except Exception as e:
                st.error(f"{u}: {e}")
        for f in up or []:
            tmp = Path(".uploads")
            tmp.mkdir(exist_ok=True)
            p = tmp / f.name
            p.write_bytes(f.getvalue())
            r = engine.ingest_file(p)
            st.success(f"{r.status}: {f.name} ({r.chunks} chunks)")

with tab_ask:
    q = st.text_input("Question", placeholder="What does ANNOY use for nearest neighbour search?")
    if st.button("Ask", type="primary") and q:
        steps = st.container()
        final = None
        for kind, payload in engine.stream(q):
            if kind == "node":
                detail = {k: v for k, v in payload.items() if k not in ("node", "ms")}
                steps.write(f"**{payload['node']}**  `{payload['ms']} ms`  {detail}")
            else:
                final = payload
        if final:
            (st.warning if final.abstained else st.success)(final.text)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Route", final.route)
            c2.metric("Support", f"{final.support:.0%}")
            c3.metric("LLM calls", final.llm_calls)
            c4.metric("Latency", f"{final.latency_s:.1f}s")
            if final.redactions:
                st.info(f"{final.redactions} instruction-like sentence(s) removed from retrieved text.")
            for c in final.citations:
                st.markdown(f"**[{c.n}]** {c.title or c.source}  \n> {c.quote}")

with tab_bench:
    res = Path("benchmarks/results")
    shown = False
    for name in ("retrieval", "e2e", "injection"):
        f = res / f"{name}.json"
        if f.exists():
            shown = True
            data = json.loads(f.read_text(encoding="utf-8"))
            st.subheader(name)
            if name == "retrieval":
                st.dataframe(data["modes"])
            elif name == "e2e":
                st.dataframe({k: data[k]["summary"] for k in ("baseline", "full")})
            else:
                st.json(data["summary"])
    if not shown:
        st.info("Run `python -m imskos eval all` to generate benchmark results.")
