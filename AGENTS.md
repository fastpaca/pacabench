# AI Agent Instructions for agentbench

> **Target Audience**: AI coding agents (Cursor, Copilot, Aider, etc.)

## Critical: Code Quality Gates

### Run Ruff on EVERY Change

Before marking any task complete, you MUST run these commands:

```bash
uv run ruff check agentbench/ runners/ --fix
uv run ruff format agentbench/ runners/
uv run ruff check agentbench/ runners/
```

**If ruff fails, the task is not complete.** Fix all issues before proceeding.

**CI/CD**: GitHub Actions will automatically run ruff on PRs.

## Quick Reference

### Project Commands

```bash
# Code quality
uv run ruff check agentbench/ runners/ --fix  # Fix issues
uv run ruff format agentbench/ runners/       # Format code
uv run ruff check agentbench/ runners/        # Check only

# Run evaluation (MemBench QA baseline)
uv run agentbench --dataset membench --runner qa/long_context --model gpt-4o-mini --limit 10

# Quick smoke test
uv run agentbench --dataset membench --runner qa/long_context --model gpt-4o-mini --limit 2

# Run GAIA agentic evaluation
uv run agentbench --dataset gaia --runner agentic/mem0 --model gpt-4o --limit 2 --split level1

# Install dependencies (all extras)
uv sync --all-extras
```

### File Structure

```
agentbench/
├── cli.py             # CLI, dataset routing, metrics display
├── proxy.py           # FastAPI LLM proxy (token/latency/cost tracking)
├── runner.py          # Subprocess runner & JSON protocol
├── datasets.py        # Dataset loaders (MemBench, LongMemEval, GAIA)
├── evaluators.py      # QA/agentic evaluation helpers
├── metrics.py         # CaseResult + aggregate/save helpers
└── runners/           # Standalone CLIs executed per case
    ├── qa/
    └── agentic/
```

## Code Style Rules (Enforced by Ruff)

### 1. Type Hints Required
```python
# ✅ Good
def compute_metrics(cases: list[CaseResult]) -> dict[str, float]:
    return {"p50": 0.5}

# ❌ Bad - missing type hints
def compute_metrics(cases):
    return {"p50": 0.5}
```

### 2. No Redundant Comments
```python
# ❌ Bad - comment just restates code
# Load dataset
cases = load_membench(limit=limit)

# ✅ Good - no comment needed, code is self-explanatory
cases = load_membench(limit=limit)
```

### 3. Extract Magic Numbers
```python
# ❌ Bad
timeout = 300.0

# ✅ Good
_REQUEST_TIMEOUT_SECONDS = 300.0
timeout = _REQUEST_TIMEOUT_SECONDS
```

### 4. Direct Attribute Access
```python
# ❌ Bad - unnecessary getattr for known fields
duration = getattr(result, "runner_duration_ms", 0)

# ✅ Good - direct access, will fail fast if field missing
duration = result.runner_duration_ms
```

### 5. No Pointless Abstraction
```python
# ❌ Bad - dictionary lookup to get a number back
PERCENTILES = {"p50": 0.50}
p50 = _calculate_percentile(values, PERCENTILES["p50"])

# ✅ Good - just use the number
p50 = _calculate_percentile(values, 0.50)
```

## Common Modifications

### Adding a Dataset Loader

Edit `agentbench/datasets.py`:

```python
def load_new_dataset(... ) -> list[Case]:
    cases = [...]
    return cases
```

Then register it inside `_load_dataset()` in `agentbench/cli.py` and expose any new CLI options if needed. Keep splits explicit (e.g., `"eval"`, `"validation"`) and return task metadata required by downstream evaluators.

### Adding or Updating Runners

Runners live under `runners/{qa|agentic}/` and are executed as standalone CLIs.

1. Create a new Python script (e.g., `runners/qa/my_runner.py`).
2. Follow the stdin/stdout JSON protocol used by existing runners.
3. Read configuration from environment (`MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`).
4. Route **all** LLM calls through the proxy (`OpenAI(base_url=...)`) so latency/cost metrics stay accurate.
5. Return `{"result": "...", "error": null}` style payloads.

### Modifying Metrics Display

Update `_print_metrics_table()` in `agentbench/cli.py` when adding/removing displayed metrics. Never drop latency rows (avg/p50/p95) or token/cost visibility. Aggregation lives in `agentbench/metrics.py`; extend `CaseResult`/`AggregatedMetrics` first, then plumb fields into the table.

## Testing Your Changes

### Minimal Test
```bash
uv run agentbench --dataset membench --runner qa/long_context --model gpt-4o-mini --limit 2
```

### GAIA Agentic Test
```bash
uv run agentbench --dataset gaia --runner agentic/mem0 --model gpt-4o --limit 2 --split level1
```

### Verify Output
```bash
# Check metrics structure
cat runs/*/metrics.json | jq .

# Verify latency metrics are non-zero (in milliseconds)
cat runs/*/metrics.json | jq '.avg_llm_latency_ms, .p50_llm_latency_ms, .p95_llm_latency_ms'
```

## Architecture Patterns

### Process-Based Runners + Proxy
- `agentbench.cli` spawns `ProxyServer` (FastAPI) to intercept OpenAI traffic.
- `spawn_runner()` executes a runner script per case using JSON stdin/stdout.
- Proxy metrics accumulate per case and are flushed after each result (`proxy.metrics.clear_metrics("_current")`).

### Metrics Collection
- Capture runner duration (`runner_duration_ms`) plus proxy metrics (`llm_latency_ms`, token counts, cost).
- `CaseResult` stores evaluator outcomes (`f1_score`, `judge_passed`) so `aggregate_results()` can derive accuracy, precision, percentiles, and judge token totals.

### Results Schema

**results.jsonl** (per-case):
```json
{
  "case_id": "string",
  "passed": true,
  "output": "model output",
  "error": null,
  "runner_duration_ms": 1234.0,
  "llm_metrics": {"llm_call_count": 2, "llm_latency_ms": [530.0, 410.0]},
  "f1_score": 0.84,
  "f1_passed": true,
  "judge_passed": true,
  "judge_metrics": {"input_tokens": 750, "output_tokens": 120}
}
```

**metrics.json** (aggregated):
```json
{
  "accuracy": 0.85,
  "precision": 0.82,
  "total_cases": 100,
  "p50_duration_s": 1.2,
  "p95_duration_s": 2.1,
  "avg_llm_latency_ms": 550.0,
  "p50_llm_latency_ms": 510.0,
  "p95_llm_latency_ms": 760.0,
  "total_input_tokens": 500000,
  "total_output_tokens": 52000,
  "total_cost_usd": 12.34,
  "total_judge_input_tokens": 8500
}
```

## Debugging

### Issue: Latency shows 0.0 ms
- Ensure runners call the proxy URL (`OPENAI_BASE_URL`) for every OpenAI request.
- Check that `proxy.metrics.get_metrics("_current")` is used when building the `CaseResult`.

### Issue: Ruff failing
```bash
uv run ruff check agentbench/ runners/        # Inspect failures
uv run ruff check agentbench/ runners/ --fix  # Auto-fix
uv run ruff format agentbench/ runners/       # Format
```

### Issue: Missing dependencies
```bash
uv sync --all-extras
```

## Performance Notes

- Default proxy port: 8000 (keep free or update `ProxyServer`/runner env).
- HTTP concurrency + timeouts inherit from the OpenAI Python client; adjust `ProxyServer` initialization if stricter limits or retries are required.
- Runner subprocess duration drives duration metrics—long tool chains will inflate `runner_duration_ms`, so keep expensive work minimal.
- One proxy per evaluation run; avoid spawning additional agents inside runners unless required.

## Non-Negotiables

1. 🚨 **ALWAYS** run ruff before completing tasks
2. 🚨 **NEVER** remove latency metrics from CLI or JSON outputs
3. 🚨 **NEVER** add redundant comments
4. 🚨 **NEVER** skip type hints
5. 🚨 **NEVER** commit code that doesn't pass `ruff check`

## Workflow

1. Make your code changes
2. Run `uv run ruff check agentbench/ runners/ --fix`
3. Run `uv run ruff format agentbench/ runners/`
4. Verify `uv run ruff check agentbench/ runners/` passes
5. Test with `uv run agentbench --dataset membench --runner qa/long_context --limit 2`
6. Verify latency metrics appear in `runs/*/metrics.json`
7. Task complete ✅

---

**Remember**: AgentBench prioritizes simplicity, explicit control of runners, and accurate performance data. When in doubt, keep abstractions thin and ensure every LLM call flows through the proxy for auditable metrics.
