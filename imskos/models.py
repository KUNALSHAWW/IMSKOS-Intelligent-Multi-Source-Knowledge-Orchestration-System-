"""Plain data types shared across the package."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source: str            # URL, file path or "wikipedia:<title>"
    title: str = ""
    idx: int = 0           # position inside the source document
    doc_hash: str = ""     # sha256 of the full source text (used for idempotent re-ingestion)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class Citation:
    n: int                 # 1-based passage number used in the answer text
    source: str
    title: str
    chunk_id: str
    quote: str             # best supporting sentence from the passage
    support: float         # share of the answer sentence's content words found in the passage


@dataclass
class Answer:
    question: str
    text: str
    route: str = ""                    # "vectorstore" or "web"
    abstained: bool = False
    grounded: bool = False
    support: float = 0.0               # share of answer sentences supported by the retrieved passages
    citations: list[Citation] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    latency_s: float = 0.0
    rewrites: int = 0
    redactions: int = 0                # instruction-like sentences removed from retrieved passages

    def to_dict(self) -> dict:
        return asdict(self)
