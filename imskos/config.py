"""Runtime settings, read from environment variables (prefix ``IMSKOS_``)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434", "gemma4:e4b", ""),
}


@dataclass
class Settings:
    provider: str = "groq"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = ""                    # e.g. cross-encoder/ms-marco-MiniLM-L-6-v2; empty disables
    store: str = "local"                      # local | astra
    store_dir: str = "kb_store"
    retrieval_mode: str = "hybrid"            # dense | bm25 | hybrid
    top_k: int = 5
    chunk_chars: int = 900
    chunk_overlap: int = 150
    max_rewrites: int = 2
    max_regenerations: int = 1
    router: str = "none"                      # none (retrieval first) | evidence | titles
    grade: bool = True                        # LLM relevance grading of retrieved passages
    verify: bool = True                       # deterministic grounding check after generation
    defense: bool = True                      # redact instruction-like sentences in retrieved text
    web_fallback: bool = True
    support_threshold: float = 0.5
    astra_endpoint: str = ""
    astra_token: str = ""
    astra_collection: str = "imskos"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        base, model, key_env = _PROVIDERS.get(self.provider, ("", "", ""))
        self.base_url = self.base_url or base
        self.model = self.model or model
        if not self.api_key and key_env:
            self.api_key = os.getenv(key_env, "")

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        def env(name, default, cast=str):
            raw = os.getenv(f"IMSKOS_{name.upper()}")
            if raw is None or raw == "":
                return default
            if cast is bool:
                return raw.strip().lower() in ("1", "true", "yes", "on")
            return cast(raw)

        d = cls()
        kw = {
            "provider": env("provider", d.provider),
            "model": env("model", ""),
            "base_url": env("base_url", ""),
            "api_key": env("api_key", ""),
            "embedding_model": env("embedding_model", d.embedding_model),
            "rerank_model": env("rerank_model", d.rerank_model),
            "store": env("store", d.store),
            "store_dir": env("store_dir", d.store_dir),
            "retrieval_mode": env("retrieval_mode", d.retrieval_mode),
            "top_k": env("top_k", d.top_k, int),
            "max_rewrites": env("max_rewrites", d.max_rewrites, int),
            "router": env("router", d.router),
            "grade": env("grade", d.grade, bool),
            "verify": env("verify", d.verify, bool),
            "defense": env("defense", d.defense, bool),
            "web_fallback": env("web_fallback", d.web_fallback, bool),
            "astra_endpoint": os.getenv("ASTRA_DB_API_ENDPOINT", ""),
            "astra_token": os.getenv("ASTRA_DB_APPLICATION_TOKEN", ""),
            "astra_collection": env("astra_collection", d.astra_collection),
        }
        kw.update(overrides)
        return cls(**kw)
