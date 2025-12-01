# Claude Instructions for PacaBench

Project-specific context for PacaBench. See AGENTS.md for universal Python patterns and Google Style Guide conventions.

## Project Overview

PacaBench is a benchmark harness for evaluating LLM agents. It runs agents as subprocesses, proxies OpenAI API calls to track metrics (latency, tokens, cost), and aggregates results.

## Commands

```bash
# Code quality (required before completing any task)
uv run ruff check pacabench/ --fix
uv run ruff format pacabench/
uv run ruff check pacabench/

# Run benchmark
uv run pacabench run --limit 10

# View results
uv run pacabench                        # Latest run summary
uv run pacabench show                   # List all runs
uv run pacabench show <run-id>          # Specific run details
uv run pacabench show <run-id> --cases  # Case-level results

# Other commands
uv run pacabench retry <run-id>         # Retry failures
uv run pacabench export <run-id>        # Export to JSON
uv run pacabench init                   # Create new project
```

## File Structure

```
pacabench/
  cli/
    app.py              # Typer CLI commands
    formatters.py       # Rich output formatting
  engine/
    orchestrator.py     # Main execution loop
    proxy.py            # FastAPI proxy for OpenAI traffic
    datasets/           # Dataset loaders (local, git, huggingface)
    evaluators/         # Evaluation logic (exact match, f1, llm judge)
    runners/            # Agent subprocess execution
  models/
    case.py             # Case, CaseResult, RunnerOutput
    config.py           # BenchmarkConfig, AgentConfig, DatasetConfig
    metrics.py          # LLMMetrics, AggregatedMetrics
    run.py              # RunMetadata, RunSummary, EvalContext
  run_manager/
    manager.py          # Run state and persistence
    storage.py          # Results file I/O
    discovery.py        # Run discovery and resolution
```

## Core Models

All models are in `pacabench/models/`. Use them for all serialization.

| Model | Purpose | Location |
|-------|---------|----------|
| `Case` | Input to agent | case.py |
| `CaseResult` | Evaluation output | case.py |
| `RunnerOutput` | Raw agent response | case.py |
| `RunMetadata` | Persisted run state | run.py |
| `RunSummary` | Lightweight run view | run.py |
| `BenchmarkConfig` | Loaded from YAML | config.py |
| `LLMMetrics` | Token/cost tracking | metrics.py |

## Architecture

### Proxy-Based Metrics

All agent OpenAI calls must go through the proxy (via `OPENAI_BASE_URL` env var). The proxy tracks:
- Call count, latency, tokens, cost per case
- Accumulated in `LLMMetrics` model

### Runner Protocol

Agents communicate via stdin/stdout JSON:

```
Input:  {"case_id": "...", "input": "...", "history": [...], "metadata": {...}}
Output: {"output": "...", "error": null}
```

### Run Persistence

Each run creates a directory under `runs/`:

| File | Model | Format |
|------|-------|--------|
| `metadata.json` | `RunMetadata` | JSON |
| `pacabench.yaml` | `BenchmarkConfig` | YAML |
| `results.jsonl` | `CaseResult` | JSONL |
| `system_errors.jsonl` | (untyped) | JSONL |

## Testing Changes

```bash
cd examples/membench_qa_test
uv run pacabench run --limit 2
uv run pacabench show <run-id> --cases
```

## Key Constraints

1. Never remove latency/token/cost metrics from output
2. All OpenAI calls must go through the proxy
3. Runners must respect the stdin/stdout JSON protocol
4. Run ruff before completing any task
5. Use models from `pacabench/models/` for all serialization
