"""Turn URLs and files into clean text, then into overlapping, sentence-aligned chunks."""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import httpx

from .models import Chunk
from .textutil import sentences

_UA = {"User-Agent": "IMSKOS/2.0 (+https://github.com/KUNALSHAWW)"}
_DROP = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"]


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, text). Prefers <article> or <main>, drops page chrome."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(_DROP):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    for br in root.find_all("br"):
        br.replace_with("\n")
    for blk in root.find_all(["p", "li", "h1", "h2", "h3", "h4", "pre", "tr"]):
        blk.append("\n")
    return title, _tidy(root.get_text())


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return _tidy("\n".join((p.extract_text() or "") for p in reader.pages))


def _tidy(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def fetch_url(url: str, timeout: float = 30.0) -> tuple[str, str]:
    r = httpx.get(url, headers=_UA, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    if "pdf" in r.headers.get("content-type", "") or url.lower().endswith(".pdf"):
        return Path(url).name, pdf_to_text(r.content)
    return html_to_text(r.text)


def load_file(path: str | Path) -> tuple[str, str]:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return p.stem, pdf_to_text(p.read_bytes())
    raw = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() in (".html", ".htm"):
        title, text = html_to_text(raw)
        return title or p.stem, text
    return p.stem, _tidy(raw)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Pack whole sentences into chunks of at most ``size`` characters.

    The last sentences of a chunk (up to ``overlap`` characters) open the next one, so a fact that
    straddles a boundary is still retrievable. A single sentence longer than ``size`` is kept whole.
    """
    sents = sentences(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        if cur and cur_len + len(s) + 1 > size:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            n = 0
            for t in reversed(cur):
                if n + len(t) > overlap:
                    break
                tail.insert(0, t)
                n += len(t) + 1
            cur, cur_len = tail, n
        cur.append(s)
        cur_len += len(s) + 1
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks


def make_chunks(source: str, title: str, text: str, size: int = 900, overlap: int = 150) -> list[Chunk]:
    doc_hash = sha256(text)
    out = []
    for i, c in enumerate(chunk_text(text, size, overlap)):
        cid = hashlib.sha1(f"{source}|{i}|{c}".encode()).hexdigest()[:16]
        out.append(Chunk(id=cid, text=c, source=source, title=title, idx=i, doc_hash=doc_hash))
    return out
