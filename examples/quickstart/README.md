# Quick Start

Two questions, one agent, metrics proxy on. This is the README Quick Start.

```bash
export OPENAI_API_KEY=sk-...
pacabench run
```

The agent calls `gpt-4o-mini` through `OPENAI_BASE_URL`. PacaBench sets that variable to a local proxy, which records latency and tokens. A `gpt-4o-mini` judge scores the answers.

Python 3 standard library only. `python` must be on `PATH`. In a pipe, add `--no-tui`.

The offline echo suite in [`examples/smoke_test`](../smoke_test/) is the contributor and CI check. It does not call a model.
