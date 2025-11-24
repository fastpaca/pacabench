# Claude Instructions for pacabench

This document provides guidelines for Claude (and other LLMs) working on the AgentBench codebase.

## Project Overview

**AgentBench** is a process-based benchmark harness for evaluating LLM memory systems and agentic workflows. It runs runners as standalone subprocesses, proxies all OpenAI API calls to track latency/tokens/cost, and aggregates precision + pass/fail metrics for QA and GAIA-style tasks.

## Code Quality Standards

### Always Run Ruff

**CRITICAL**: Before completing any code changes, ALWAYS run ruff to ensure code quality:

```bash
# Auto-fix and format
uv run ruff check pacabench/ runners/ --fix
uv run ruff format pacabench/ runners/

# Verify everything passes
uv run ruff check pacabench/ runners/ && echo "✅ Ready to commit"
```

Never deliver code without running ruff first. GitHub Actions also runs ruff on every PR.

### Code Style Guidelines

1. **No redundant comments** – the code should be self-explanatory.
   - ❌ `# Load dataset` followed by `cases = load_membench()`
   - ✅ Explain *why* when a comment is required, e.g., `# Split once to reuse eval/val across runs`

2. **No unnecessary abstractions** – prefer direct, explicit code paths.
   - ❌ `PERCENTILES = {"p50": 0.50}`
   - ✅ `np.percentile(durations, 50)`

3. **No magic numbers** – extract constants and name them clearly.
   - ❌ `proxy = ProxyServer(port=9000)`
   - ✅ `_PROXY_PORT = 8000`

4. **Direct attribute access** – do not use `getattr`/`dict.get` for required fields.
   - ❌ `duration = getattr(result, "runner_duration_ms", 0)`
   - ✅ `duration = result.runner_duration_ms`

5. **Type hints everywhere** – every function must specify parameter and return types.
   - ✅ `def aggregate_results(results: list[CaseResult]) -> AggregatedMetrics:`

### Circular Dependency Prevention

**CRITICAL**: Never use `if TYPE_CHECKING:` guards – this is a code smell indicating poor architecture. Always fix circular dependencies architecturally.

#### Rule 1: All Protocols and Data Models Go in types.py

`types.py` is the foundation - it has NO pacabench imports and defines all core types and protocols.

**✅ Correct Pattern:**
```python
# types.py - Pure types, no pacabench imports
from __future__ import annotations
from typing import Protocol

class RunnerContext(BaseModel):
    """Lightweight context for runner execution - only runtime config."""
    model: str
    proxy_port: int
    openai_api_key: str
    embedding_model: str | None

class Runner(Protocol):
    """Protocol for runners that execute test cases."""
    async def run_case(self, case: Case, ctx: RunnerContext) -> RunnerOutput:
        ...
```

**Why This Works:**
1. Runner Protocol references `RunnerContext` - both defined in `types.py`
2. No circular dependencies because `types.py` is self-contained
3. Pipeline can import both `Runner` and `RunnerContext` from `types.py`
4. Clean dependency hierarchy: `types.py` → everything else

#### Rule 2: Don't Use "Context" as a Dumping Ground

**❌ BAD - God Object Anti-Pattern:**
```python
class EvalContext(BaseModel):
    dataset: Dataset      # ← Already a parameter
    runner: Runner        # ← Already a parameter
    results: Results      # ← Already created locally
    model: str            # ← Already a parameter
    proxy: ProxyServer    # ← Already created locally
    # ... 10 more fields
```

This is just parameter passing with extra steps. Pass parameters directly!

**✅ GOOD - Direct Parameter Passing:**
```python
async def run_case(
    case: Case,
    runner: Runner,
    runner_ctx: RunnerContext,  # ← Only what runners need
    dataset: Dataset,
    proxy: ProxyServer,
    judge_model: str,
    judge_client: AsyncOpenAI,
) -> CaseResult:
    ...
```

**Adding New Runners:**

New runners are registered in `cli.py`'s `RUNNERS` dict:
```python
RUNNERS = {
    "qa/long_context": LongContextRunner(),
    "qa/mem0": Mem0Runner(),
    "agentic/long_context": LongContextAgenticRunner(),
    "agentic/mem0": Mem0AgenticRunner(),
}
```

## Project Structure

```
pacabench/
├── cli.py              # Typer CLI, dataset loader, metrics table
├── datasets.py         # MemBench, LongMemEval, GAIA loaders
├── evaluators.py       # QA / agentic evaluators
├── proxy.py            # FastAPI proxy tracking latency/tokens/cost
├── runner.py           # Subprocess runner for QA/agentic scripts
├── metrics.py          # CaseResult + AggregatedMetrics helpers
└── runners/
    ├── qa/
    └── agentic/
```

## Key Architectural Patterns

### 1. Proxy-Orchestrated Evaluations
```python
proxy = ProxyServer(port=8000, openai_api_key=openai_api_key)
proxy.start()
llm_metrics = proxy.metrics.get_metrics("_current")
proxy.metrics.clear_metrics("_current")
```
All runners must hit the proxy (via `OPENAI_BASE_URL=http://localhost:8000/v1`) so metrics are complete.

### 2. Runner Protocol
```python
# stdin -> Case JSON
case = json.loads(sys.stdin.read())

# stdout <- {"result": str | None, "error": str | None}
print(json.dumps({"result": answer, "error": None}))
```
The harness enforces this contract and will surface `runner_result.error` if anything fails.

### 3. Metrics Aggregation
- Build `CaseResult` instances with runner duration, proxy stats, and evaluator outputs (`f1_score`, `judge_passed`).
- `aggregate_results()` computes accuracy, precision, duration percentiles, LLM latency percentiles, and cost.
- `_print_metrics_table()` surfaces the aggregated view—never remove latency/token/cost rows.

## Common Tasks

### Adding a Dataset

1. Implement a loader in `pacabench/datasets.py` that returns `list[Case]`.
2. Wire it up inside `_load_dataset()` in `pacabench/cli.py` with explicit split handling.
3. Include any metadata needed by evaluators (choices, GAIA files, etc.).

### Adding a Runner

1. Create `runners/qa/new_runner.py` or `runners/agentic/new_runner.py`.
2. Read `MODEL`, `OPENAI_API_KEY`, and `OPENAI_BASE_URL` from the environment.
3. Send all OpenAI calls through `OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))`.
4. Respect stdin/stdout protocol and return structured JSON.
5. Update docs/tests to mention the new runner.

### Updating Metrics or CLI Output

1. Extend `CaseResult`/`AggregatedMetrics` in `pacabench/metrics.py`.
2. Persist the new fields in `save_results()`.
3. Update `_print_metrics_table()` to display them (without removing latency metrics).

## Testing Changes

```bash
# Run QA baseline on a few cases
uv run pacabench --dataset membench --runner qa/long_context --model gpt-4o-mini --limit 2

# Run GAIA agentic check
uv run pacabench --dataset gaia --runner agentic/mem0 --model gpt-4o --limit 2 --split level1

# Inspect metrics
cat runs/*/metrics.json | jq '.accuracy, .p50_llm_latency_ms'
```

## Pre-Commit Checklist

- [ ] `uv run ruff check pacabench/ runners/ --fix`
- [ ] `uv run ruff format pacabench/ runners/`
- [ ] `uv run ruff check pacabench/ runners/`
- [ ] `uv run pacabench --dataset membench --runner qa/long_context --limit 2`
- [ ] Verify latency metrics (avg/p50/p95) are non-zero in `runs/*/metrics.json`
- [ ] Confirm all functions include type hints and no redundant comments were added

## Performance Considerations

1. **Proxy lifecycle** – one proxy per evaluation. Ensure `.start()`/`.stop()` remain symmetric.
2. **Port usage** – default is 8000; runners must respect `OPENAI_BASE_URL`.
3. **Runner duration** – `runner_duration_ms` includes every tool call. Keep heavy work minimal or parallelize.
4. **LLM token accounting** – `MetricsCollector` relies on OpenAI usage objects. Use official clients, not raw HTTP.

## Don't Break These Rules

1. ❌ Never commit without running ruff.
2. ❌ Never remove latency/token/cost metrics from output.
3. ❌ Never bypass the proxy with direct OpenAI calls.
4. ❌ Never add pointless abstraction layers.
5. ✅ Always keep type hints and direct attribute access.
6. ✅ Always test changes with actual evaluation runs (at least a two-case smoke test).

## Questions?

Prefer **simplicity** over cleverness, **explicit** code over implicit magic, and keep performance observable by routing everything through the proxy.
