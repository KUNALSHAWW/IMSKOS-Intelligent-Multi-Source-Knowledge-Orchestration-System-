# Benchmarks

Raw results: `benchmarks/results/` (`retrieval.json`, `e2e.json`, `e2e_none.json`, `injection.json`). Each file records the SHA-256 of the corpus text used.

Setup: `gemma4:e4b` through Ollama on CPU, `all-MiniLM-L6-v2` embeddings, optional `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, three Lilian Weng posts (agents, prompt engineering, adversarial attacks) cut into 165 chunks of about 900 characters. The posts are fetched at run time and are not redistributed here.

## Questions

`imskos/eval/data/questions.json`: 30 answerable questions (each with lowercase gold phrases that must occur in a relevant chunk and answer phrases that must occur in a correct answer), 10 unanswerable questions about the same topics, 10 general-knowledge questions. Gold phrases are verified against the live corpus before every run; the run stops if one is missing.

## Retrieval

| Mode | hit@1 | hit@3 | hit@5 | MRR@10 (95% CI) | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|
| Dense | 0.700 | 0.867 | 0.933 | 0.810 (0.70 to 0.91) | 0.842 | 20 ms |
| BM25 | 0.767 | 0.967 | 1.000 | 0.868 (0.78 to 0.95) | 0.897 | 0.5 ms |
| Hybrid (RRF) | 0.767 | 1.000 | 1.000 | 0.878 (0.79 to 0.95) | 0.891 | 16 ms |
| Hybrid + rerank | 0.967 | 1.000 | 1.000 | 0.983 (0.95 to 1.00) | 0.977 | 2,084 ms |

- Hybrid does not beat BM25 on hit@1 here. The questions share vocabulary with the gold passages, which favours lexical search. Hybrid does reach hit@3 of 1.00.
- The cross-encoder is the largest single improvement (hit@1 0.77 to 0.97), and by far the most expensive step: 2 s per query on CPU while the machine was loaded.
- Chunk size sweep (hybrid): 500 characters gives hit@1 0.80 and MRR 0.892, 900 gives 0.77 and 0.878, 1,500 gives 0.63 and 0.782. The confidence intervals overlap for 500 and 900. The default stays at 900 rather than being tuned on the benchmark.

## End to end

Correct means the answer contains a gold answer phrase and is not an abstention. Citation hit means a cited chunk contains the gold evidence.

| | Plain RAG | Agent, LLM router | Agent, retrieval first |
|---|---|---|---|
| Answerable, correct | 100% | 76.7% | 80.0% |
| Answerable, abstained | 0% | 20.0% | 13.3% |
| Citation hit | 100% | 87.5% | 88.5% |
| Unanswerable, abstained | 100% | 100% | 90% |
| General questions, correct | not supported | 90% | 80% |
| Routing accuracy | n/a | 85% | n/a (no router) |
| Mean model calls | 1.0 | 2.4 | 3.2 |
| p50 latency | 66 s | 51 s | 37 s |

**What happened.**

- The LLM router sent 5 of 30 knowledge-base questions to Wikipedia. The agent then abstained on all five. That is why retrieval-first routing became the default. This change was made after seeing the first run, so the retrieval-first numbers are not held-out.
- In the retrieval-first run the agent abstained on four in-corpus questions after grading or verification, and answered two more with partial or off-target sentences. The grounding check is strict about paraphrase, and a small model paraphrases.
- One unanswerable question (about the size of a human preference dataset) got an answer built from a passage sentence on a nearby topic. The lexical check accepted it because the sentence was supported by its passage, even though it did not answer the question.
- Plain RAG answered everything it could and abstained on everything it should. On this model and corpus it is the most accurate configuration. The agent is the only configuration that can answer the general questions, and it is the one that carries the safety machinery.
- Latency was measured while other processes shared the CPU. It is not comparable across columns.

## Indirect prompt injection

6 attack templates x 8 questions x defence off and on, with a canary string (`PWNED-7731`). Templates 1 to 4 are instructions to the model ("ignore all previous instructions", "note to the AI assistant", "new instructions", "from now on"). Templates 5 and 6 are content poisoning (a false "official" answer with no instruction).

| | Instruction-style, defence off | Instruction-style, defence on | Poisoning, off | Poisoning, on |
|---|---|---|---|---|
| Attacks that succeeded | 0 / 32 | 0 / 32 | 0 / 16 | 0 / 16 |
| Answers still correct | 87.5% | 87.5% | 81.3% | 81.3% |

- The sanitiser fired in all 32 instruction-style runs with the defence on (the poisoned sentence was retrieved and removed) and in none of the benign runs. On the three-post corpus it removed no sentences.
- **No attack succeeded even with the defence off.** The model, the "passages are data" system prompt, and the grounding check together resisted all of them, so this benchmark cannot show a benefit from the sanitiser. The defence is verified by unit tests with a deliberately obedient fake model, where it takes the attack success rate from 100% to 0% for instruction-style attacks and does nothing for content poisoning. A weaker model, or a stronger attack, would be needed to measure a real effect.

## Reproducing

```bash
python -m imskos eval retrieval                                   # about a minute, no LLM
python -m imskos eval e2e --provider ollama --model gemma4:e4b --router evidence   # baseline + agent, writes e2e.json
python -m imskos eval e2e --provider ollama --model gemma4:e4b --router none --configs full   # writes e2e_none.json
python -m imskos eval injection --provider ollama --model gemma4:e4b
```
