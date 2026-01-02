# GAIA Benchmark Example

This example demonstrates running the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) with PacaBench.

GAIA (General AI Assistants) evaluates AI assistants on real-world tasks requiring web browsing, file handling, and multi-step reasoning.

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Authenticate with HuggingFace

GAIA is a gated dataset. You need to:
1. Accept the terms at https://huggingface.co/datasets/gaia-benchmark/GAIA
2. Login locally:

```bash
huggingface-cli login
```

### 3. Prepare the dataset

Download and convert the GAIA dataset to PacaBench format:

```bash
# All validation cases (165 questions)
uv run python prepare_dataset.py

# Or filter by difficulty level
uv run python prepare_dataset.py --level 1  # Easiest (54 cases)
uv run python prepare_dataset.py --level 2  # Medium (86 cases)
uv run python prepare_dataset.py --level 3  # Hardest (25 cases)
```

This creates `data/validation.jsonl` and downloads file attachments to `data/files/`.

### 4. Set your API key

The agents use OpenAI models:

```bash
export OPENAI_API_KEY=sk-...
```

## Running the benchmark

```bash
# Run with PacaBench
pacabench run

# Or limit to N cases for testing
pacabench run --limit 5
```

## Agent

`smolagent_agent.py` uses the [smolagents](https://github.com/huggingface/smolagents) framework with:
- DuckDuckGo web search
- Web page reading
- Python code execution

This demonstrates the PacaBench agent protocol: read JSONL from stdin, write `{"output": "..."}` to stdout.

## Expected results

GAIA is a challenging benchmark. State-of-the-art agents typically achieve:
- Level 1: 50-70% accuracy
- Level 2: 30-50% accuracy
- Level 3: 10-30% accuracy

The smolagent in this example achieves roughly 30-40% on Level 1 questions.
