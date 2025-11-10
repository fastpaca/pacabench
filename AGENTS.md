# AI Agent Instructions for memharness

> **Target Audience**: AI coding agents (Cursor, Copilot, Aider, etc.)

## Critical: Code Quality Gates

### Run Ruff on EVERY Change

Before marking any task complete, you MUST run these commands:

```bash
uv run ruff check memharness/ --fix
uv run ruff format memharness/
uv run ruff check memharness/
```

**If ruff fails, the task is not complete.** Fix all issues before proceeding.

**CI/CD**: GitHub Actions will automatically check ruff on PRs.

## Quick Reference

### Project Commands

```bash
# Code quality
uv run ruff check memharness/ --fix  # Fix issues
uv run ruff format memharness/       # Format code
uv run ruff check memharness/        # Check only

# Run evaluation
uv run memharness -d membench -c claude-sonnet-long-context -l 10 -v

# Quick test
uv run memharness -d membench -c gpt-4o-mini-long-context -l 2

# List available datasets
uv run memharness --list-datasets

# List available models
uv run memharness --list-configs

# Install dependencies
uv sync
```

### File Structure

```
memharness/
├── cli.py                    # CLI & metrics display (DO NOT remove latency metrics!)
├── configs.py                # Add new models/datasets here
├── answerers/
│   └── long_context.py       # LLM task implementation
└── datasets/
    └── membench.py           # Dataset loaders
```

## Code Style Rules (Enforced by Ruff)

### 1. Type Hints Required
```python
# ✅ Good
def compute_metrics(cases: list) -> dict[str, float]:
    return {"p50": 0.5}

# ❌ Bad - missing type hints
def compute_metrics(cases):
    return {"p50": 0.5}
```

### 2. No Redundant Comments
```python
# ❌ Bad - comment just restates code
# Load dataset
dataset = DATASETS[dataset_name](limit=limit)

# ✅ Good - no comment needed, code is self-explanatory
dataset = DATASETS[dataset_name](limit=limit)
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
duration = getattr(case, "task_duration", 0)

# ✅ Good - direct access, will fail fast if field missing
duration = case.task_duration
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

### Adding a Model Configuration

Edit `configs.py`:

```python
ANSWERERS = {
    # Add your config
    "my-model-name": long_context_answerer(
        model="model-identifier",
        provider="openai",  # or "anthropic"
        base_url="http://...",  # optional for local models
        api_key="...",  # optional
    ),
}
```

Then run ruff.

### Adding a Dataset

Edit `configs.py`:

```python
from functools import partial

DATASETS = {
    "my-dataset": load_membench,  # if using default args
    "my-dataset-variant": partial(load_membench, agent_type="ThirdAgent"),
}
```

Then run ruff.

### Modifying Metrics Display

Edit `cli.py` → `_create_metrics_table()` function.

**WARNING**: Do NOT remove latency metrics. Performance measurement is critical.

## Testing Your Changes

### Minimal Test
```bash
uv run memharness -d membench -c gpt-4o-mini-long-context -l 2
```

### Full Test
```bash
uv run memharness -d membench -c claude-sonnet-long-context -l 10 -v
```

### Verify Output
```bash
# Check that metrics.json exists and has correct structure
cat runs/membench-*/metrics.json | jq .

# Verify latency metrics are non-zero
cat runs/membench-*/metrics.json | jq '.p50_latency_s, .p95_latency_s'
```

## Architecture Patterns

### Lazy Agent Initialization

Agents are expensive to create. Use closure pattern:

```python
def long_context_answerer(...):
    _agent = None

    def get_agent() -> Agent:
        nonlocal _agent
        if _agent is not None:
            return _agent
        _agent = Agent(...)  # Initialize once
        return _agent

    async def task(inputs: dict) -> str:
        agent = get_agent()  # Reuse across calls
        ...
```

### Metrics Collection

- **Tokens**: Use `increment_eval_metric("input_tokens", count)`
- **Latency**: Automatically tracked by pydantic-evals
- **Accuracy**: Computed from `case.assertions`

### Results Schema

**results.jsonl** (per-case):
```json
{
  "case_id": "string",
  "output": "string",
  "expected": "string",
  "correct": true,
  "task_duration_s": 1.23,
  "total_duration_s": 1.45,
  "metrics": {"input_tokens": 1000, "output_tokens": 50},
  "metadata": {}
}
```

**metrics.json** (aggregated):
```json
{
  "accuracy": 0.85,
  "total_cases": 100,
  "p50_latency_s": 1.2,
  "p95_latency_s": 2.1,
  "p99_latency_s": 3.4,
  "avg_input_tokens": 5000,
  ...
}
```

## Debugging

### Issue: Latency shows 0.00s

Check that you're accessing the correct attributes:
- ✅ `case.task_duration`
- ✅ `case.total_duration`
- ❌ `case.duration_s` (doesn't exist in pydantic-evals)

### Issue: Ruff failing

```bash
uv run ruff check memharness/        # See what's wrong
uv run ruff check memharness/ --fix  # Auto-fix
uv run ruff format memharness/       # Format
```

### Issue: Import errors

```bash
# Reinstall dependencies
uv sync
```

## Performance Notes

- Default concurrency: 10 (use `-j` to adjust)
- HTTP connections: 20 max (see `_MAX_CONNECTIONS`)
- Request timeout: 300s (see `_REQUEST_TIMEOUT_SECONDS`)
- Agent caching: One agent per evaluation run (via closure)

## Non-Negotiables

1. 🚨 **ALWAYS** run ruff before completing tasks
2. 🚨 **NEVER** remove latency metrics from output
3. 🚨 **NEVER** add redundant comments
4. 🚨 **NEVER** skip type hints
5. 🚨 **NEVER** commit code that doesn't pass `ruff check`

## Workflow

1. Make your code changes
2. Run `uv run ruff check memharness/ --fix`
3. Run `uv run ruff format memharness/`
4. Verify `uv run ruff check memharness/` passes
5. Test with `uv run memharness -d membench -c <config> -l 2`
6. Verify latency metrics appear in output
7. Task complete ✅

---

**Remember**: This codebase values simplicity, performance, and correctness. When in doubt, prefer explicit, straightforward code over clever abstractions.
