"""Defence against indirect prompt injection through retrieved documents.

Anything a retriever returns is untrusted text: a web page or PDF can contain sentences that
address the model instead of the reader ("ignore all previous instructions and ..."). Before a
passage reaches the prompt, sentences that look like instructions to the model are removed and
counted. The remaining text stays usable, so a blog post that merely *discusses* jailbreaks still
answers questions about them. Passages are also wrapped as data in the prompt (see ``graph.py``).

This is a heuristic filter, not a proof of safety. ``imskos.eval.injection`` measures how much it
helps against a fixed attack set, and states what it does not catch.
"""
from __future__ import annotations

import re

from .textutil import sentences

_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any|your|the)\b[^.\n]{0,40}\b(instruction|prompt|rule|context|direction|guideline)s?\b",
    r"\bnew (instruction|task|directive)s?\s*:",
    r"\b(reply|respond|answer|output|say|write)\b[^.\n]{0,20}\b(only|just|exactly)\b[^.\n]{0,12}[\"'“]",
    r"\b(reply|respond|answer|output|say)\s+(only|just|exactly)\s+with\b",
    r"\b(reply|respond|answer|output)\s+with\s+(the\s+)?(word|phrase|text|string)\b",
    r"\bfrom now on\b[^.\n]{0,60}\b(you|your|answer|respond|reply)\b",
    r"\byou are (now|no longer)\b",
    r"\b(reveal|print|show|repeat|leak|disclose)\b[^.\n]{0,30}\b(system prompt|your prompt|your instructions|api key|secret|password)s?\b",
    r"\bdo not (mention|tell|reveal|cite|say)\b[^.\n]{0,40}\b(user|this|these|source|instruction)s?\b",
    r"<\s*\|?\s*(system|im_start|assistant)\s*\|?\s*>",
    r"\b(assistant|ai|model|llm|chatbot)\s*[,:]\s*(ignore|disregard|you must|you should|please)\b",
    r"\bwhen (asked|answering|summari[sz]ing)\b[^.\n]{0,60}\b(always|instead|must)\b",
]
_RX = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

REDACTED = "[removed: instruction-like text]"


def is_injection(sentence: str) -> bool:
    return any(rx.search(sentence) for rx in _RX)


def scan(text: str) -> list[str]:
    """Return the instruction-like sentences found in ``text``."""
    return [s for s in sentences(text) if is_injection(s)]


def redact(text: str) -> tuple[str, int]:
    """Replace instruction-like sentences by a marker. Returns (clean_text, number_removed)."""
    bad = scan(text)
    for s in bad:
        text = text.replace(s, REDACTED)
    return text, len(bad)
