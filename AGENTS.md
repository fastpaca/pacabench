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
├── cli.py             # CLI wrapper around library
├── pipeline.py        # Main evaluation pipeline orchestration
├── proxy.py           # FastAPI LLM proxy (token/latency/cost tracking)
├── context.py         # EvalContext (shared state)
├── config.py          # RunConfig model
├── datasets/          # Dataset abstractions and loaders
│   ├── base.py        # Abstract Dataset class
│   ├── qa_dataset.py  # QA dataset implementation
│   ├── gaia_dataset.py # GAIA dataset implementation
│   ├── membench.py    # MemBench loader
│   ├── longmemeval.py # LongMemEval loader
│   └── gaia.py        # GAIA loader
├── runners/           # Runner abstractions
│   ├── base.py        # Runner protocol
│   ├── command.py     # CommandRunner (subprocess runner)
│   └── resolver.py    # Runner spec resolution
├── stages/            # Pipeline stages
│   ├── case.py        # Case model
│   ├── runner.py      # RunnerOutput model & legacy runner function
│   ├── evaluator.py   # Evaluation functions
│   └── result.py      # CaseResult, AggregatedMetrics, aggregation
└── runners/           # Standalone runner scripts executed per case
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

### Adding a Dataset

Datasets own both loading and evaluation strategy:

1. **Create a loader function** (e.g., in `agentbench/datasets/my_dataset.py`):
   ```python
   def load_my_dataset(limit: int | None = None) -> list[Case]:
       # Load and return Case objects
       return cases
   ```

2. **Create a Dataset subclass** (e.g., `agentbench/datasets/my_dataset.py`):
   ```python
   from agentbench.datasets.base import Dataset
   from agentbench.datasets.qa_dataset import QaDataset  # or create custom
   
   # For QA-style datasets, reuse QaDataset:
   my_dataset = QaDataset(name="my_dataset", loader_func=load_my_dataset)
   ```

3. **Register in `agentbench/datasets/__init__.py`**:
   ```python
   def get_dataset(dataset: DatasetEnum | str, ...) -> Dataset:
       # Add your dataset to the registry
   ```

The Dataset's `evaluate_case()` method defines how correctness is judged. QA datasets use F1 + LLM judge; GAIA uses LLM-as-judge only.

### Adding or Updating Runners

Runners can be implemented in two ways:

#### 1. External Command Runners (Any Language)

Runners are standalone scripts that follow a simple JSON protocol:

1. **Built-in runners**: Place scripts under `runners/{qa|agentic}/` with naming pattern `{name}_runner.py` (e.g., `runners/qa/long_context_runner.py`). Reference them via shorthand: `qa/long_context`.

2. **External runners**: Place scripts anywhere on the filesystem. Reference them via:
   - Relative path: `./path/to/my_runner.py`
   - Absolute path: `/abs/path/to/my_runner.py`

3. **Runner Protocol**:
   - Read `Case` JSON from stdin: `{"id": "...", "task_type": "...", "inputs": {...}}`
   - Read configuration from environment:
     - `MODEL`: Model name
     - `OPENAI_API_KEY`: API key
     - `OPENAI_BASE_URL`: Proxy base URL (always route LLM calls here for metrics)
     - `AGENTBENCH_RUN_ID`: Run identifier
     - `AGENTBENCH_DATASET`: Dataset name
   - Write JSON to stdout: `{"result": "...", "error": null}` or `{"result": null, "error": "..."}`

#### 2. Python Library Runners

For Python users, implement the `Runner` protocol:

```python
from agentbench.runners.base import Runner
from agentbench.stages.case import Case
from agentbench.stages.runner import RunnerOutput
from agentbench.context import EvalContext

class MyRunner:
    async def run_case(self, case: Case, ctx: EvalContext) -> RunnerOutput:
        # Your implementation
        return RunnerOutput(result="...", error=None, duration_ms=123.0)
```

Use `CommandRunner` for external command execution, or implement custom logic directly.

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

### Library-First Design
- **Core models** (`Case`, `RunnerOutput`, `EvaluationOutput`, `CaseResult`, `AggregatedMetrics`) are Pydantic BaseModel subclasses for validation and serialization.
- **Datasets** own loading (`load_cases()`) and evaluation (`evaluate_case()`).
- **Runners** implement the `Runner` protocol; `CommandRunner` handles external processes.
- **Pipeline** (`pipeline.run()`) orchestrates: load dataset → run cases → evaluate → aggregate.

### Process-Based Runners + Proxy
- `agentbench.pipeline` spawns `ProxyServer` (FastAPI) to intercept OpenAI traffic.
- `CommandRunner` executes runner scripts per case using JSON stdin/stdout protocol.
- Proxy metrics accumulate per case and are flushed after each result (`proxy.metrics.clear_metrics("_current")`).

### Runner Spec Resolution
- Built-in shorthand (e.g., `qa/long_context`) → `runners/{spec}_runner.py` in project root.
- Filesystem paths (`./script.py`, `/abs/path.py`) → resolved directly, no renaming.
- Centralized resolver (`runners/resolver.py`) handles all path logic.

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
