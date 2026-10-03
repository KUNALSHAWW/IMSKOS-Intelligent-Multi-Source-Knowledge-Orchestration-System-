"""Smoke tests for the Streamlit app. They need no network: hash embedder, local provider, patched fetch."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    st.cache_resource.clear()
    monkeypatch.setenv("IMSKOS_EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("IMSKOS_STORE_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("IMSKOS_PROVIDER", "ollama_cloud")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("IMSKOS_MODEL", "deepseek-v3.1:671b")
    monkeypatch.delenv("IMSKOS_SEED_URLS", raising=False)


def test_sidebar_defaults_come_from_the_environment():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.selectbox[0].value == "ollama_cloud"
    assert at.sidebar.text_input[0].value == "deepseek-v3.1:671b"
    assert len(at.tabs) == 3


def test_empty_store_is_seeded_from_urls(monkeypatch):
    monkeypatch.setenv("IMSKOS_SEED_URLS", "https://example.org/a, https://example.org/b")
    monkeypatch.setattr("imskos.engine.fetch_url", lambda url: (f"Doc {url[-1]}", f"Fact number {url[-1]} about vector stores. " * 5))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    cards = " ".join(m.value for m in at.markdown)
    assert '<div class="k">Documents</div><div class="v">2</div>' in cards
