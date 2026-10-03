"""The benchmark corpus: three Lilian Weng posts, fetched at run time and cached locally.

The posts are not redistributed in this repository. Each run records the SHA-256 of the text it
used, so a result can be tied to the exact corpus version it was measured on.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..loaders import fetch_url, sha256

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def load_corpus(cache_dir: str | Path = "corpus_cache") -> list[tuple[str, str, str]]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    docs = []
    for url in URLS:
        f = cache / (url.rstrip("/").rsplit("/", 1)[-1] + ".json")
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
        else:
            title, text = fetch_url(url)
            d = {"url": url, "title": title, "text": text}
            f.write_text(json.dumps(d), encoding="utf-8")
        docs.append((d["url"], d["title"], d["text"]))
    return docs


def corpus_fingerprint(docs: list[tuple[str, str, str]]) -> dict[str, str]:
    return {url: sha256(text) for url, _, text in docs}
