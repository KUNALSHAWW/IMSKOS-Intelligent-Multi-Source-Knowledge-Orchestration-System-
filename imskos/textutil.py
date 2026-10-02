"""Tokenisation and sentence helpers used by BM25, the hash embedder and the grounding check."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\"'])|\n+")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her his how i if in into is it
    its may might more most of on or our she should so some such than that the their them then there these they this to
    was we were what when where which while who whom why will with would you your also not no yes about over under
    between each other any both""".split()
)


def stem(w: str) -> str:
    """Very light suffix stripping so 'agents' matches 'agent' and 'planning' matches 'plan'."""
    for suf in ("ing", "ies", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if keep_stop or w not in STOPWORDS:
            out.append(stem(w))
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]
