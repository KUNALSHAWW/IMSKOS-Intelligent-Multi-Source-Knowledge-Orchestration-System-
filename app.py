"""Streamlit UI for IMSKOS: ask with a live agent trace, manage the knowledge base, inspect benchmarks."""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from imskos import Engine, Settings
from imskos import ui_theme as ui

ACCENT = "#7C83FF"
ui.apply(ACCENT, "IMSKOS", "◆")

EXAMPLES = [
    "What does ANNOY use as its core data structure?",
    "How does self-consistency sampling pick the final answer?",
    "What is the capital of Australia?",
    "What learning rate was used to fine-tune Toolformer?",
]


@st.cache_resource(show_spinner="Loading models...")
def get_engine(provider: str, model: str, mode: str, grade: bool, verify: bool, defense: bool, web: bool) -> Engine:
    return Engine(Settings.from_env(provider=provider, model=model, retrieval_mode=mode, grade=grade,
                                    verify=verify, defense=defense, web_fallback=web))


with st.sidebar:
    st.markdown("### IMSKOS")
    st.caption("Agentic RAG that grades, verifies and abstains.")
    provider = st.selectbox("LLM provider", ["groq", "openai", "ollama"], format_func=lambda p: {"groq": "Groq", "openai": "OpenAI", "ollama": "Ollama (local)"}[p])
    model = st.text_input("Model (blank for default)", "")
    mode = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"])
    st.markdown("**Agent behaviour**")
    grade = st.toggle("Grade retrieved passages", True)
    verify = st.toggle("Verify grounding", True)
    defense = st.toggle("Redact injected instructions", True)
    web = st.toggle("Wikipedia fallback", True)

ui.hero(
    "Agentic RAG",
    "Answers you can check, or no answer at all.",
    "A LangGraph agent that routes, retrieves with hybrid search, grades what it found, rewrites weak queries, "
    "verifies every sentence against its sources and abstains when it cannot ground an answer.",
    [("Hybrid BM25 + dense", "accent"), ("Grounding check", "ok"), ("Injection defence", "warn"), ("Live trace", "neutral")],
)

try:
    engine = get_engine(provider, model, mode, grade, verify, defense, web)
except Exception as e:  # missing API key, Ollama not running, model download blocked
    ui.banner("engine error", f"Could not start the engine: {e}", "bad")
    st.stop()

tab_ask, tab_kb, tab_bench = st.tabs(["Ask", "Knowledge base", "Benchmarks"])

with tab_ask:
    clicked = None
    cols = st.columns(len(EXAMPLES))
    for c, ex_q in zip(cols, EXAMPLES):
        if c.button(ex_q, use_container_width=True, key=ex_q):
            clicked = ex_q
    q = st.chat_input("Ask a question about the indexed documents or general knowledge") or clicked
    if q:
        with st.status("Agent running", expanded=True) as status:
            final = None
            for kind, payload in engine.stream(q):
                if kind == "node":
                    detail = ", ".join(f"{k}={v}" for k, v in payload.items() if k not in ("node", "ms", "raw"))
                    st.markdown(f"`{payload['node']}` &nbsp; {detail} &nbsp; *{payload['ms']:.0f} ms*")
                else:
                    final = payload
            status.update(label="Done", state="complete", expanded=False)
        st.session_state["final"] = final

    final = st.session_state.get("final")
    if final:
        ui.banner("abstained" if final.abstained else ("grounded" if final.grounded else "unverified"),
                  final.text, "warn" if final.abstained else ("ok" if final.grounded else "bad"))
        ui.cards([("Route", final.route or "n/a", ""), ("Support", f"{final.support:.0%}", "sentences backed by sources"),
                  ("LLM calls", str(final.llm_calls), f"{final.completion_tokens} completion tokens"),
                  ("Latency", f"{final.latency_s:.1f}s", "")])
        if final.redactions:
            ui.banner("defence", f"{final.redactions} instruction-like sentence(s) removed from retrieved text.", "warn")
        if final.citations:
            ui.section("Sources", "Each quote is the sentence that best supports the answer.")
            for c in final.citations:
                ui.quote(f"[{c.n}] {c.title or c.source}", c.quote)
        ui.section("Agent trace", "One row per graph node.")
        ui.steps([(t["node"], ", ".join(f"{k}={v}" for k, v in t.items() if k not in ("node", "ms", "raw")), f"{t['ms']:.0f} ms")
                  for t in final.trace])

with tab_kb:
    stat = engine.status()
    ui.cards([("Indexed chunks", str(stat["chunks"]), stat["store"] + " store"), ("Documents", str(len(stat["sources"])), ""),
              ("Embedder", stat["embedder"].split("/")[-1], "")])
    for title in stat["titles"]:
        st.markdown(f"- {title}")
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

with tab_bench:
    res = Path("benchmarks/results")
    shown = False
    r = res / "retrieval.json"
    if r.exists():
        shown = True
        d = json.loads(r.read_text(encoding="utf-8"))
        ui.section("Retrieval", f"{d['questions']} labelled questions over {d['chunks']} chunks. A chunk is relevant if it contains the gold phrase.")
        modes = list(d["modes"])
        fig = go.Figure()
        for metric, color in (("hit@1", ACCENT), ("hit@5", "#3DD68C"), ("mrr@10", "#F5B93E")):
            fig.add_bar(name=metric, x=modes, y=[d["modes"][m][metric] for m in modes], marker_color=color)
        st.plotly_chart(ui.style_fig(fig, ACCENT, 340).update_layout(barmode="group").update_yaxes(range=[0, 1.05], tickformat=".0%"),
                        use_container_width=True)
    e, e2 = res / "e2e.json", res / "e2e_none.json"
    if e.exists():
        shown = True
        d = json.loads(e.read_text(encoding="utf-8"))
        configs = {"Plain RAG": d["baseline"]["summary"], "Agent, LLM router": d["full"]["summary"]}
        if e2.exists():
            configs["Agent, retrieval first"] = json.loads(e2.read_text(encoding="utf-8"))["full"]["summary"]
        ui.section("End to end", f"Real model ({d['llm']}), 30 answerable and 10 unanswerable questions. Differences of a few points are within noise: one question is 3.3 points.")
        metrics = {"answerable_correct": "Answered correctly", "citation_hit": "Citation hits the evidence",
                   "unanswerable_abstained": "Abstained when unanswerable"}
        fig = go.Figure()
        for name, color in zip(configs, ("#6B7078", ACCENT, "#3DD68C")):
            fig.add_bar(name=name, x=list(metrics.values()), y=[configs[name][k] for k in metrics], marker_color=color)
        st.plotly_chart(ui.style_fig(fig, ACCENT, 340).update_layout(barmode="group").update_yaxes(range=[0, 1.05], tickformat=".0%"),
                        use_container_width=True)
        ui.cards([(n, f"{c['latency_s_p50']:.0f}s p50", f"{c['llm_calls_mean']:.1f} model calls per question") for n, c in configs.items()])
    i = res / "injection.json"
    if i.exists():
        shown = True
        d = json.loads(i.read_text(encoding="utf-8"))
        sm = d["summary"]
        ui.section("Indirect prompt injection", "6 attack templates x 8 questions, defence off and on.")
        ui.cards([("Instruction-style, defence off", f"{sm['defense_off']['instruction_style']['attack_success_rate']:.0%}", "attack success"),
                  ("Instruction-style, defence on", f"{sm['defense_on']['instruction_style']['attack_success_rate']:.0%}", "attack success"),
                  ("Content poisoning, defence on", f"{sm['defense_on']['content_poisoning']['attack_success_rate']:.0%}", "attack success")])
    if not shown:
        st.info("Run `python -m imskos eval all` to generate benchmark results.")
