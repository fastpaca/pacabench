# Claude Instructions for memharness

This document provides guidelines for Claude (or other LLMs) working on the memharness codebase.

## Project Overview

**memharness** is a memory QA benchmark harness for evaluating LLM memory systems. It runs evaluations using pydantic-evals and tracks both accuracy and performance metrics (latency, tokens).

## Code Quality Standards

### Always Run Ruff

**CRITICAL**: Before completing any code changes, ALWAYS run ruff to ensure code quality:

```bash
# Auto-fix and format
uv run ruff check memharness/ --fix
uv run ruff format memharness/

# Verify everything passes
uv run ruff check memharness/ && echo "✅ Ready to commit"
```

**Never** deliver code without running ruff first. This is non-negotiable.

**CI/CD**: GitHub Actions automatically runs ruff on all PRs and pushes to main.

### Code Style Guidelines

1. **No redundant comments** - Don't write comments that just restate what the code does
   - ❌ `# Load dataset` followed by `dataset = load_membench()`
   - ✅ `# Split by agent type to enable comparative benchmarking`

2. **No unnecessary abstractions** - Don't create indirection without clear value
   - ❌ `PERCENTILES = {"p50": 0.50}` then `PERCENTILES["p50"]`
   - ✅ Just use `0.50` directly

3. **No magic numbers** - Extract constants with meaningful names
   - ❌ `timeout=300.0`
   - ✅ `_REQUEST_TIMEOUT_SECONDS = 300.0`

4. **Direct attribute access** - Don't use `getattr` for known required fields
   - ❌ `getattr(case, "task_duration", 0)`
   - ✅ `case.task_duration`

5. **Type hints everywhere** - All functions must have parameter and return types
   - ✅ `def compute_metrics(cases: list) -> dict[str, float]:`

## Project Structure

```
memharness/
├── cli.py              # Main CLI entry point
├── configs.py          # Dataset and model configurations
├── answerers/
│   └── long_context.py # Task implementations
└── datasets/
    └── membench.py     # Dataset loaders
```

## Key Architectural Patterns

### 1. Factory Pattern for Answerers
```python
def long_context_answerer(model: str, provider: str) -> Callable:
    """Returns a task function with lazy-initialized agent."""
    _agent = None

    def get_agent() -> Agent:
        nonlocal _agent
        if _agent is not None:
            return _agent
        # Initialize once, reuse across calls
        _agent = Agent(...)
        return _agent

    async def task(inputs: dict) -> str:
        agent = get_agent()
        result = await agent.run(...)
        return str(result.output)

    return task
```

### 2. Metrics Collection
- **Token metrics**: Use `increment_eval_metric()` from pydantic-evals
- **Latency metrics**: Automatically tracked by pydantic-evals as `task_duration` and `total_duration`
- **Percentiles**: Calculate using linear interpolation, not simple indexing

### 3. Results Output
- **config.json**: Run metadata
- **results.jsonl**: Per-case results with durations and metrics
- **metrics.json**: Aggregated stats including p50/p95/p99 latency
- **Console**: Rich table with metrics (always shown) + optional verbose report

## Common Tasks

### Adding a New Model Provider

1. Update `long_context.py` to handle the new provider:
```python
elif provider == "new_provider":
    model_obj = NewProviderModel(model, api_key=api_key)
```

2. Add configuration in `configs.py`:
```python
ANSWERERS = {
    "new-model": long_context_answerer(
        model="model-name",
        provider="new_provider",
    ),
}
```

3. Run tests and ruff before committing

### Adding a New Dataset

1. Create loader in `datasets/`:
```python
def load_new_dataset(limit: int | None = None) -> Dataset:
    cases = [Case(name=..., inputs=..., expected_output=...) for ...]
    return Dataset(cases=cases, evaluators=[...])
```

2. Register in `configs.py`:
```python
DATASETS = {
    "new-dataset": load_new_dataset,
}
```

### Modifying CLI Output

- Metrics table: Edit `_create_metrics_table()` in `cli.py`
- Add new metrics: Compute in `_compute_*_metrics()` functions
- Always preserve latency metrics - they are critical

## Testing Changes

```bash
# Run evaluation with verbose output
uv run memharness -d membench -c claude-sonnet-long-context -l 5 -v

# Run quick test
uv run memharness -d membench -c gpt-4o-mini-long-context -l 2

# Check results
cat runs/membench-*-*/metrics.json
```

## Pre-Commit Checklist

- [ ] Run `uv run ruff check memharness/ --fix`
- [ ] Run `uv run ruff format memharness/`
- [ ] Verify `uv run ruff check memharness/` passes
- [ ] Test with `uv run memharness -d membench -c <config> -l 2`
- [ ] Check that latency metrics show non-zero values
- [ ] Ensure no redundant comments were added
- [ ] Verify all functions have type hints

**Note**: GitHub Actions will automatically verify ruff passes on your PR.

## Performance Considerations

1. **Concurrency**: Default is 10, increase for faster evals (use `-j` flag)
2. **HTTP Connection Limits**: Set in `_MAX_CONNECTIONS` constant
3. **Timeouts**: LLM requests timeout at `_REQUEST_TIMEOUT_SECONDS` (300s)
4. **Lazy Loading**: Agents are initialized once per evaluation run

## Don't Break These Rules

1. ❌ Never commit without running ruff
2. ❌ Never remove latency metrics from output
3. ❌ Never use `getattr` for pydantic-evals dataclass fields
4. ❌ Never add pointless abstraction layers
5. ❌ Never write comments that just restate code
6. ✅ Always maintain type hints
7. ✅ Always test changes with actual evaluation runs
8. ✅ Always preserve backward compatibility for saved results format

## Questions?

If you're unsure about an architectural decision, prefer:
- **Simplicity** over abstraction
- **Explicitness** over magic
- **Direct code** over clever indirection
- **Performance** matters - measure latency!
