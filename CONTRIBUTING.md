# Contributing

```bash
pip install -r requirements-dev.txt && pip install -e .
make lint test
```

- Tests use a hash embedder and a scripted fake LLM, so they need no network or model. Add a scripted graph test for any new node or edge.
- Benchmark claims in the docs must come from `benchmarks/results/*.json`. Re-run the suite and commit the JSON; do not edit numbers by hand.
- Changes to the grounding check or the sanitiser need a test with a sentence that should pass and one that should not.
- Keep commits small and describe the behaviour change.
