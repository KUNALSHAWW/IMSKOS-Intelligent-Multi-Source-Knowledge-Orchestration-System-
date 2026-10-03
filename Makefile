.PHONY: install test lint ingest-demo ask eval-retrieval eval eval-injection app serve clean

PY ?= python
PROVIDER ?= ollama
MODEL ?= gemma4:e4b

install:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

test:
	$(PY) -m pytest

lint:
	ruff check .

ingest-demo:                ## index the three Lilian Weng posts used by the benchmark
	$(PY) -m imskos ingest https://lilianweng.github.io/posts/2023-06-23-agent/ https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/ https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/

ask:
	$(PY) -m imskos ask "$(Q)" --trace

eval-retrieval:             ## no LLM needed: dense vs BM25 vs hybrid vs rerank
	$(PY) -m imskos eval retrieval

eval:                       ## retrieval + end-to-end + injection (needs an LLM)
	$(PY) -m imskos eval all --provider $(PROVIDER) --model $(MODEL)

eval-injection:
	$(PY) -m imskos eval injection --provider $(PROVIDER) --model $(MODEL)

app:
	streamlit run app.py

serve:
	$(PY) -m imskos serve

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info kb_store corpus_cache
