"""REST API. ``/ask/stream`` is Server-Sent Events: one event per graph node, then the final answer."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .engine import Engine


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class IngestIn(BaseModel):
    urls: list[str] = Field(default_factory=list, max_length=20)
    text: str | None = Field(default=None, max_length=2_000_000)
    source: str = "api:text"
    title: str = ""


def create_app(engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="IMSKOS", version="2.0.0")
    state = {"engine": engine}
    keys = {k for k in os.getenv("IMSKOS_API_KEYS", "").split(",") if k}

    def eng() -> Engine:
        if state["engine"] is None:
            state["engine"] = Engine()
        return state["engine"]

    def auth(x_api_key: str | None = Header(default=None)) -> None:
        if keys and x_api_key not in keys:
            raise HTTPException(401, "missing or invalid X-API-Key")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/status", dependencies=[Depends(auth)])
    def status():
        return eng().status()

    @app.post("/ingest", dependencies=[Depends(auth)])
    def ingest(body: IngestIn):
        e = eng()
        out = []
        try:
            for u in body.urls:
                out.append(asdict(e.ingest_url(u)))
            if body.text:
                out.append(asdict(e.ingest_text(body.source, body.title, body.text)))
        except Exception as ex:  # fetch/parse failures are the caller's input problem
            raise HTTPException(422, f"ingest failed: {type(ex).__name__}: {ex}") from ex
        return {"results": out, "chunks": e.store.count()}

    @app.post("/ask", dependencies=[Depends(auth)])
    def ask(body: AskIn):
        return eng().ask(body.question).to_dict()

    @app.post("/ask/stream", dependencies=[Depends(auth)])
    def ask_stream(body: AskIn):
        def events():
            for kind, payload in eng().stream(body.question):
                data = payload.to_dict() if kind == "final" else payload
                yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app()
