import numpy as np
import pytest

from imskos.embed import HashEmbedder
from imskos.loaders import chunk_text, html_to_text, make_chunks
from imskos.models import Chunk, Hit
from imskos.retrieval import Ingestor, Retriever, rrf
from imskos.store import BM25, LocalStore
from imskos.textutil import tokens

from .conftest import DOCS, make_engine


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = chunk_text(text, size=200, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 260 for c in chunks)
    # overlap: the last sentence of a chunk reappears at the start of the next one
    last = chunks[0].split(". ")[-1][:20]
    assert last in chunks[1]


def test_long_sentence_is_kept_whole():
    s = "word " * 500 + "end."
    assert chunk_text(s, size=100, overlap=10) == [s.strip()]


def test_chunk_ids_are_deterministic():
    a = make_chunks("s", "t", "One. Two. Three.", 20, 0)
    b = make_chunks("s", "t", "One. Two. Three.", 20, 0)
    assert [c.id for c in a] == [c.id for c in b]


def test_html_extraction_drops_chrome():
    html = ("<html><head><title>T</title></head><body><nav>menu menu</nav><article><h1>Head</h1>"
            "<p>Body text one.</p><p>Body two.</p></article><footer>foot</footer><script>x()</script></body></html>")
    title, text = html_to_text(html)
    assert title == "T"
    assert "Body text one." in text and "menu" not in text and "foot" not in text and "x()" not in text


def test_ingest_is_idempotent_and_updates():
    store = LocalStore()
    ing = Ingestor(store, HashEmbedder())
    assert ing.ingest_text("s", "T", "Alpha beta gamma. Delta epsilon.").status == "added"
    n = store.count()
    assert ing.ingest_text("s", "T", "Alpha beta gamma. Delta epsilon.").status == "unchanged"
    assert store.count() == n
    assert ing.ingest_text("s", "T", "Completely different content now.").status == "updated"
    assert all("Alpha" not in c.text for c in store.chunks)


def test_store_persists_and_reloads(tmp_path):
    s1 = LocalStore(tmp_path)
    Ingestor(s1, HashEmbedder()).ingest_text("s", "T", "Persistent fact about vectors.")
    s2 = LocalStore(tmp_path)
    assert s2.count() == s1.count() and s2.sources() == s1.sources()
    q = HashEmbedder().encode(["vectors"])[0]
    assert s2.dense(q, 1)[0].chunk.text.startswith("Persistent")


def test_bm25_ranks_exact_term_first():
    docs = [tokens(t) for t in ["cats and dogs", "quantization of vectors", "unrelated words here"]]
    sc = BM25(docs).scores(tokens("quantization"))
    assert int(np.argmax(sc)) == 1 and sc[0] == 0


def test_rrf_prefers_documents_ranked_by_both_lists():
    c = [Chunk(str(i), "t", "s") for i in range(4)]
    a = [Hit(c[0], 1, 1), Hit(c[1], 1, 2)]
    b = [Hit(c[2], 1, 1), Hit(c[1], 1, 2), Hit(c[3], 1, 3)]
    fused = rrf([a, b])
    assert fused[0].chunk.id == "1"          # rank 2 in both lists beats rank 1 in only one
    assert {h.chunk.id for h in fused} == {"0", "1", "2", "3"}


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_all_modes_find_the_right_chunk(mode):
    eng = make_engine(None or __import__("imskos.llm", fromlist=["FakeLLM"]).FakeLLM(["x"]))
    hits = Retriever(eng.store, eng.embedder, mode).search("which structure does ANNOY search with random trees", k=2)
    assert "ANNOY" in hits[0].chunk.text


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        Retriever(LocalStore(), HashEmbedder(), "magic")


def test_reranker_reorders():
    eng = make_engine(__import__("imskos.llm", fromlist=["FakeLLM"]).FakeLLM(["x"]))

    class Prefer:
        name = "stub"

        def score(self, q, passages):
            return [1.0 if "Reflexion" in p else 0.0 for p in passages]

    r = Retriever(eng.store, eng.embedder, "hybrid", Prefer())
    assert "Reflexion" in r.search("anything", k=3)[0].chunk.text
    assert DOCS  # corpus fixture is intact
