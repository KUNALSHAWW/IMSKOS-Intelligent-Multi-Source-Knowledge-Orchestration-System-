import json

import httpx
import numpy as np
from fastapi.testclient import TestClient

from imskos.api import create_app
from imskos.llm import FakeLLM
from imskos.models import Chunk
from imskos.store import AstraStore

from .conftest import fake_script, make_engine


def client(monkeypatch=None, keys=None):
    import os
    if keys:
        os.environ["IMSKOS_API_KEYS"] = keys
    else:
        os.environ.pop("IMSKOS_API_KEYS", None)
    return TestClient(create_app(make_engine(FakeLLM(fake_script()))))


def test_health_and_ask():
    c = client()
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/ask", json={"question": "What data structure does ANNOY use?"})
    body = r.json()
    assert r.status_code == 200 and body["grounded"] and body["citations"][0]["title"] == "LLM Agents"


def test_ask_validates_input():
    assert client().post("/ask", json={"question": "x"}).status_code == 422


def test_stream_is_sse_with_final_event():
    r = client().post("/ask/stream", json={"question": "What data structure does ANNOY use?"})
    events = [line[7:] for line in r.text.splitlines() if line.startswith("event:")]
    assert events[0] == "node" and events[-1] == "final"
    last = [line for line in r.text.splitlines() if line.startswith("data:")][-1]
    assert json.loads(last[5:])["route"] == "vectorstore"


def test_ingest_text_then_status():
    c = client()
    before = c.get("/status").json()["chunks"]
    r = c.post("/ingest", json={"text": "A new fact about vector stores.", "source": "api:test", "title": "New"})
    assert r.status_code == 200 and r.json()["results"][0]["status"] == "added"
    assert c.get("/status").json()["chunks"] == before + 1
    assert c.post("/ingest", json={"text": "A new fact about vector stores.", "source": "api:test"}).json()["results"][0]["status"] == "unchanged"


def test_api_key_required_when_configured():
    c = client(keys="k1")
    assert c.get("/status").status_code == 401
    assert c.get("/status", headers={"X-API-Key": "k1"}).status_code == 200
    assert c.get("/health").status_code == 200
    client()  # resets the env var


# ---- Astra Data API contract, against a mock transport -----------------------------------------
class FakeAstra:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.log: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.log.append({"path": request.url.path, "body": body, "token": request.headers.get("Token")})
        cmd = next(iter(body))
        if cmd == "createCollection":
            return httpx.Response(200, json={"status": {"ok": 1}})
        if cmd == "insertMany":
            for d in body[cmd]["documents"]:
                self.docs[d["_id"]] = d
            return httpx.Response(200, json={"status": {"insertedIds": list(self.docs)}})
        if cmd == "deleteMany":
            src = body[cmd]["filter"]["source"]
            n = [k for k, d in self.docs.items() if d["source"] == src]
            for k in n:
                del self.docs[k]
            return httpx.Response(200, json={"status": {"deletedCount": len(n)}})
        if cmd == "countDocuments":
            return httpx.Response(200, json={"status": {"count": len(self.docs)}})
        if cmd == "find":
            if "sort" in body[cmd]:
                q = np.array(body[cmd]["sort"]["$vector"])
                ranked = sorted(self.docs.values(), key=lambda d: -float(np.dot(q, d["$vector"])))
                out = [{**d, "$similarity": float(np.dot(q, d["$vector"]))} for d in ranked[: body[cmd]["options"]["limit"]]]
            else:
                out = list(self.docs.values())
            return httpx.Response(200, json={"data": {"documents": out, "nextPageState": None}})
        return httpx.Response(400, json={"errors": [{"message": f"unknown command {cmd}"}]})


def test_astra_store_contract():
    fake = FakeAstra()
    http = httpx.Client(transport=httpx.MockTransport(fake), headers={"Token": "tok"})
    store = AstraStore("https://db.example.com", "tok", "kb", dim=3, client=http)
    assert fake.log[0]["body"]["createCollection"]["options"]["vector"] == {"dimension": 3, "metric": "cosine"}
    chunks = [Chunk(f"c{i}", f"text {i}", "src-a" if i < 3 else "src-b", f"T{i}", i, "h") for i in range(25)]
    vecs = np.eye(3, dtype=np.float32)[[i % 3 for i in range(25)]]
    store.add(chunks, vecs)
    inserts = [m for m in fake.log if "insertMany" in m["body"]]
    assert len(inserts) == 2 and len(inserts[0]["body"]["insertMany"]["documents"]) == 20   # batched by 20
    assert store.count() == 25
    hits = store.dense(np.array([0, 1, 0], dtype=np.float32), 2)
    assert hits[0].chunk.id in {"c1", "c4"} and hits[0].score > hits[-1].score - 1e-9
    assert store.sources() == {"src-a": "h", "src-b": "h"}
    assert store.delete_source("src-a") == 3 and store.count() == 22
    assert store.lexical("anything", 3) == [] and not store.supports_lexical
