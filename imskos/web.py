"""Wikipedia fallback used when the knowledge base cannot answer."""
from __future__ import annotations

import httpx

from .loaders import make_chunks
from .models import Chunk

_API = "https://en.wikipedia.org/w/api.php"
_UA = {"User-Agent": "IMSKOS/2.0 (+https://github.com/KUNALSHAWW)"}


def wikipedia_search(query: str, pages: int = 2, max_chunks: int = 3) -> list[Chunk]:
    """Return up to ``max_chunks`` chunks taken from the introductions of the top Wikipedia pages."""
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query, "gsrlimit": pages,
        "prop": "extracts", "exintro": 1, "explaintext": 1, "exchars": 1800,
    }
    try:
        r = httpx.get(_API, params=params, headers=_UA, timeout=20)
        r.raise_for_status()
        found = sorted(r.json().get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    except (httpx.HTTPError, ValueError):
        return []
    out: list[Chunk] = []
    for p in found:
        text = (p.get("extract") or "").strip()
        if text:
            url = f"https://en.wikipedia.org/wiki/{p['title'].replace(' ', '_')}"
            out.extend(make_chunks(url, p["title"], text, size=900, overlap=0))
    return out[:max_chunks]
