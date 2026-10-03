# Architecture

## The graph

IMSKOS is a LangGraph `StateGraph` ([imskos/graph.py](../imskos/graph.py)). Each node returns a state update and a trace entry (node name, milliseconds, details), so every answer carries its own execution record.

```
START -> route (optional) --vectorstore--> retrieve -> grade --relevant--> generate -> verify --grounded--> END
            |                      ^           |                   ^            |
            |                      |           +--none relevant--> rewrite      +--not grounded--> regenerate -> generate
            +--------web--------> web <--------+ (after max rewrites)                        (after 1 retry) -> abstain
```

| Node | What it does | Model call |
|---|---|---|
| `route` | Optional LLM router (`IMSKOS_ROUTER=evidence` or `titles`). Off by default: with `none` every question is retrieved first and the grader decides whether to fall back to Wikipedia. The LLM router sent 5 of 30 knowledge-base questions to the web in the benchmark, which is why it is not the default. | only when enabled |
| `retrieve` | Hybrid search (dense + BM25, fused with reciprocal rank fusion), optional cross-encoder rerank. Instruction-like sentences are removed from the passages. | no |
| `grade` | The LLM keeps only passages that help answer the question (corrective RAG). Unparseable output keeps everything so recall is not lost. | yes |
| `rewrite` | If nothing is relevant, reformulates the query and searches again (max 2). | yes |
| `web` | Wikipedia fallback when the knowledge base cannot answer. | no |
| `generate` | Answers from numbered passages and cites them. | yes |
| `verify` | Deterministic grounding check of every answer sentence ([imskos/support.py](../imskos/support.py)). | no |
| `regenerate` / `abstain` | One retry with feedback about unsupported sentences, then an explicit "I don't know based on the available sources." | no |

## Grounding check

For each answer sentence, content words (stemmed, stop words removed) are compared with each retrieved passage:

- the sentence is supported when at least `support_threshold` (default 0.5) of its content words occur in one passage;
- every number in the sentence must occur in that passage, so "900 tools" cannot be supported by "53 tools";
- very short answers ("Canberra.") are checked as a whole;
- an answer is accepted when at least 80% of its sentences are supported and at least one citation resolves.

The best-matching passage and its best sentence become the citation and the quote shown to the user. Citation numbers the model invents are stripped.

This is a lexical test. It catches invented facts, numbers and names. It can be fooled by a fluent restatement that reuses passage words with a different meaning.

## Prompt-injection defence

Retrieved text is untrusted ([imskos/sanitize.py](../imskos/sanitize.py)). Sentences that address the model ("ignore all previous instructions", "note to the AI assistant", "from now on you answer only...") are replaced by a marker before they reach any prompt, including the router prompt. The rest of the passage stays usable. The generation prompt also tells the model that passages are data. The filter is heuristic; the injection benchmark reports what it stops and what it does not (content poisoning).

## Storage

| Backend | Search | Notes |
|---|---|---|
| `LocalStore` | dense (numpy) and BM25 | persisted to `kb_store/` (`chunks.json`, `vectors.npy`); used by the benchmarks |
| `AstraStore` | dense only | DataStax Astra DB Data API over HTTP; covered by contract tests against a mock transport, not run against a live database |

Ingestion is idempotent per source: the SHA-256 of the extracted text decides between `unchanged`, `updated` (old chunks replaced) and `added`. Chunks are sentence-aligned (900 characters, 150 overlap by default).

## Interfaces

- CLI: `python -m imskos ingest | ask | status | serve | eval`
- REST: `POST /ask`, `POST /ask/stream` (Server-Sent Events: one `node` event per graph node, then `final`), `POST /ingest`, `GET /status`, `GET /health`; optional `X-API-Key` through `IMSKOS_API_KEYS`
- Streamlit app: ingest, ask with a live trace, benchmark tab

## Testing

The test suite runs without network or model downloads: a hash embedder replaces the sentence-transformer, and a scripted `FakeLLM` drives every graph path (happy path, web route, rewrite then recover, rewrite then abstain, regenerate then fix, regenerate then abstain, invalid citations, injected passages, streaming, empty index).
