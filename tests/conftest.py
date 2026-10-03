import dataclasses

import pytest

from imskos.config import Settings
from imskos.embed import HashEmbedder
from imskos.engine import Engine
from imskos.llm import FakeLLM
from imskos.store import LocalStore

DOCS = {
    "kb://agents": ("LLM Agents", (
        "ANNOY uses random projection trees to search for nearest neighbors. "
        "HNSW builds hierarchical layers of small world graphs and the middle layers create shortcuts. "
        "ScaNN introduces anisotropic vector quantization for maximum inner product search. "
        "Reflexion keeps up to three self reflections in working memory as context for the agent.")),
    "kb://prompts": ("Prompt Engineering", (
        "Self consistency sampling draws several outputs at temperature above zero and picks the majority vote. "
        "Chain of thought prompting writes reasoning steps before the final answer. "
        "Few shot prompting shows demonstrations of input and output pairs to the model.")),
}


def fake_script(route="vectorstore", grade="[1]", answer="ANNOY uses random projection trees [1].", rewrite="annoy trees"):
    def f(messages, temperature):
        system = messages[0]["content"]
        if "You route questions" in system:
            return route
        if "You grade search results" in system:
            return grade
        if "Rewrite the question" in system:
            return rewrite
        return answer
    return f


def make_engine(llm, web=None, **overrides) -> Engine:
    s = dataclasses.replace(Settings(provider="ollama"), **overrides)
    emb = HashEmbedder()
    eng = Engine(s, llm=llm, embedder=emb, store=LocalStore(), web_search=web or (lambda q: []))
    for src, (title, text) in DOCS.items():
        eng.ingest_text(src, title, text)
    return eng


@pytest.fixture
def engine():
    return make_engine(FakeLLM(fake_script()))
