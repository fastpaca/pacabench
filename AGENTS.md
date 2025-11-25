# AI Agent Instructions for pacabench

> **Target Audience**: AI coding agents (Cursor, Copilot, Aider, etc.)

## Critical: Code Quality Gates

### Run Ruff on EVERY Change

Before marking any task complete, you MUST run these commands:

```bash
uv run ruff check pacabench/ --fix
uv run ruff format pacabench/
uv run ruff check pacabench/
```

**If ruff fails, the task is not complete.** Fix all issues before proceeding.

**CI/CD**: GitHub Actions will automatically run ruff on PRs.

## Quick Reference

### Project Commands

```bash
# Code quality
uv run ruff check pacabench/ --fix  # Fix issues
uv run ruff format pacabench/       # Format code
uv run ruff check pacabench/        # Check only

# Run benchmark (quick test)
uv run pacabench run --limit 10

# Check results
uv run pacabench              # Show latest run
uv run pacabench show         # List all runs
uv run pacabench show <run>   # Show specific run
uv run pacabench show <run> --cases     # Show case results
uv run pacabench show <run> --failures  # Show failures only

# Retry failures
uv run pacabench retry <run>

# Export results
uv run pacabench export <run>

# Install dependencies (all extras)
uv sync --all-extras
```

### File Structure

```
pacabench/
├── cli.py             # CLI wrapper around library
├── pipeline.py        # Main evaluation pipeline orchestration
├── proxy.py           # FastAPI LLM proxy (token/latency/cost tracking)
├── metrics.py         # AggregatedMetrics and aggregation logic
├── results.py         # Results container and JSON serialization
├── types.py           # Shared type definitions (Case, Runner, etc.)
├── datasets/          # Dataset abstractions and loaders
│   ├── base.py        # Abstract Dataset class
│   ├── membench.py    # MemBench loader
│   ├── longmemeval.py # LongMemEval loader
│   └── gaia.py        # GAIA loader
└── runners/           # Runner implementations
    ├── qa_long_context.py
    ├── qa_mem0.py
    ├── agentic_long_context.py
    └── agentic_mem0.py
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

1. **Create a loader function** (e.g., in `pacabench/datasets/my_dataset.py`):
   ```python
   def load_my_dataset(limit: int | None = None) -> list[Case]:
       # Load and return Case objects
       return cases
   ```

2. **Create a Dataset subclass** (e.g., `pacabench/datasets/my_dataset.py`):
   ```python
   from pacabench.datasets.base import Dataset
   from pacabench.datasets.qa_dataset import QaDataset  # or create custom
   
   # For QA-style datasets, reuse QaDataset:
   my_dataset = QaDataset(name="my_dataset", loader_func=load_my_dataset)
   ```

3. **Register in `pacabench/datasets/__init__.py`**:
   ```python
   def get_dataset(dataset: DatasetEnum | str, ...) -> Dataset:
       # Add your dataset to the registry
   ```

The Dataset's `evaluate_case()` method defines how correctness is judged. QA datasets use F1 + LLM judge; GAIA uses LLM-as-judge only.

### Adding or Updating Runners

Runners can be implemented in two ways:

#### 1. External Command Runners

*Not currently implemented in CLI, but architecture supports it.*


#### 2. Python Library Runners

For Python users, implement the `Runner` protocol:

```python
from pacabench.runners.base import Runner
from pacabench.stages.case import Case
from pacabench.stages.runner import RunnerOutput
from pacabench.context import EvalContext

class MyRunner:
    async def run_case(self, case: Case, ctx: EvalContext) -> RunnerOutput:
        # Your implementation
        return RunnerOutput(result="...", error=None, duration_ms=123.0)
```

Use `CommandRunner` for external command execution, or implement custom logic directly.

### Modifying Metrics Display

Update `_print_metrics_table()` in `pacabench/cli.py` when adding/removing displayed metrics. Never drop latency rows (avg/p50/p95) or token/cost visibility. Aggregation lives in `pacabench/metrics.py`; extend `CaseResult`/`AggregatedMetrics` first, then plumb fields into the table.

## Testing Your Changes

### Minimal Test
```bash
cd examples/membench_qa_test
uv run pacabench run --limit 2 --agents long-context-baseline
```

### Verify Output
```bash
# Check latest run
uv run pacabench

# View cases
uv run pacabench show <run-id> --cases

# Export and inspect
uv run pacabench export <run-id> | jq .
```

## Architecture Patterns

### Library-First Design
- **Core models** (`Case`, `RunnerOutput`, `EvaluationOutput`, `CaseResult`, `AggregatedMetrics`) are Pydantic BaseModel subclasses for validation and serialization.
- **Datasets** own loading (`load_cases()`) and evaluation (`evaluate_case()`).
- **Runners** implement the `Runner` protocol; `CommandRunner` handles external processes.
- **Pipeline** (`pipeline.run()`) orchestrates: load dataset → run cases → evaluate → aggregate.

### Process-Based Runners + Proxy
- `pacabench.pipeline` spawns `ProxyServer` (FastAPI) to intercept OpenAI traffic.
- `CommandRunner` executes runner scripts per case using JSON stdin/stdout protocol.
- Proxy metrics accumulate per case and are flushed after each result (`proxy.metrics.clear_metrics("_current")`).

### Runner Spec Resolution
- Built-in shorthand (e.g., `qa/long_context`) → Resolved via `RUNNERS` map in `pacabench/cli.py`.
- Maps to runner implementations in `pacabench/runners/`.

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
uv run ruff check pacabench/        # Inspect failures
uv run ruff check pacabench/ --fix  # Auto-fix
uv run ruff format pacabench/       # Format
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
2. Run `uv run ruff check pacabench/ --fix`
3. Run `uv run ruff format pacabench/`
4. Verify `uv run ruff check pacabench/` passes
5. Test in examples folder: `cd examples/membench_qa_test && uv run pacabench run --limit 2`
6. Verify results: `uv run pacabench show <run-id> --cases`
7. Task complete ✅

---

**Remember**: PacaBench prioritizes simplicity, explicit control of runners, and accurate performance data. When in doubt, keep abstractions thin and ensure every LLM call flows through the proxy for auditable metrics.
