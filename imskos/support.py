"""Deterministic grounding check.

For every sentence of an answer, measure how many of its content words occur in the retrieved
passages. The best-matching passage becomes the citation, and its best sentence becomes the quote
shown to the user. No model call is involved, so the check is fast, reproducible and testable.

It is a lexical test: it catches invented facts, numbers and names well, and it can be fooled by a
fluent restatement that reuses passage words while changing their meaning. The benchmark reports
how it behaves on real model output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Citation, Hit
from .textutil import sentences, tokens

ABSTAIN_TEXT = "I don't know based on the available sources."
_ABSTAIN_RX = re.compile(r"\b(i )?(do not|don't|do n't|cannot|can't) (know|find|answer)\b|\bnot (enough|sufficient) information\b"
                         r"|\bno (relevant )?(information|answer)\b|\bpassages? (do|does) not\b", re.IGNORECASE)
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN_RX.search(text)) and len(text) < 220


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text)


def clean_citations(text: str, n_passages: int) -> str:
    """Drop citation markers that point at a passage that does not exist."""
    def fix(m: re.Match) -> str:
        keep = [x for x in re.split(r"\s*,\s*", m.group(1)) if 1 <= int(x) <= n_passages]
        return f"[{', '.join(keep)}]" if keep else ""
    return _CITE.sub(fix, text)


def _ratio(sent_tokens: list[str], passage_tokens: set[str]) -> float:
    return sum(t in passage_tokens for t in sent_tokens) / len(sent_tokens)


@dataclass
class Analysis:
    support: float                                   # share of checked sentences that are supported
    unsupported: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)


def analyze(answer: str, hits: list[Hit], threshold: float = 0.5, min_tokens: int = 3) -> Analysis:
    ptoks = [set(tokens(h.chunk.text)) for h in hits]
    checked = supported = 0
    unsupported: list[str] = []
    cites: dict[int, Citation] = {}
    plain = strip_citations(answer)
    units = [s for s in sentences(plain) if len(tokens(s)) >= min_tokens]
    if not units and tokens(plain):                 # a very short answer such as "Jupiter." is checked as a whole
        units = [plain]
    for sent in units if hits else []:
        st = tokens(sent)
        checked += 1
        nums = [t for t in st if any(ch.isdigit() for ch in t)]
        # a number the passage does not contain disqualifies that passage: "900 tools" is not "53 tools"
        scores = [_ratio(st, p) if all(n in p for n in nums) else 0.0 for p in ptoks]
        best = max(range(len(hits)), key=scores.__getitem__)
        if scores[best] >= threshold:
            supported += 1
            h = hits[best]
            quote = max(sentences(h.chunk.text), key=lambda s: _ratio(st, set(tokens(s))), default=h.chunk.text)
            prev = cites.get(best + 1)
            if prev is None or scores[best] > prev.support:
                cites[best + 1] = Citation(best + 1, h.chunk.source, h.chunk.title, h.chunk.id, quote[:400], round(scores[best], 3))
        else:
            unsupported.append(sent)
    share = supported / checked if checked else 0.0
    return Analysis(round(share, 3), unsupported, sorted(cites.values(), key=lambda c: c.n))
