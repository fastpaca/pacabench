import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from agentbench.analysis import print_report
from agentbench.config import load_config
from agentbench.core import Harness

app = typer.Typer()


@app.command()
def run(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to configuration file")
    ] = Path("agentbench.yaml"),
    limit: Annotated[
        int | None, typer.Option("--limit", "-l", help="Limit number of cases per dataset")
    ] = None,
    concurrency: Annotated[int | None, typer.Option(help="Override concurrency")] = None,
    agents: Annotated[
        str | None, typer.Option(help="Comma-separated list of agents to run")
    ] = None,
):
    """Execute a benchmark run."""
    if not config.exists():
        typer.echo(f"Config file not found: {config}")
        raise typer.Exit(code=1)

    cfg = load_config(config)

    # Overrides
    if concurrency:
        cfg.config.concurrency = concurrency

    if agents:
        agent_names = [a.strip() for a in agents.split(",")]
        cfg.agents = [a for a in cfg.agents if a.name in agent_names]
        if not cfg.agents:
            typer.echo(f"No agents found matching: {agents}")
            raise typer.Exit(code=1)

    harness = Harness(cfg)
    asyncio.run(harness.run(limit=limit))


@app.command()
def init():
    """Initialize a new AgentBench project."""
    # Check if agentbench.yaml exists
    if os.path.exists("agentbench.yaml"):
        typer.echo("agentbench.yaml already exists.")
        return

    # Create dummy config
    content = """name: my-benchmark
description: Example benchmark
version: "0.1.0"

config:
  concurrency: 2
  timeout_seconds: 60
  proxy:
    enabled: true
    provider: "openai"

agents:
  - name: "example-agent"
    command: "python agent.py"
    env:
      OPENAI_API_KEY: "${OPENAI_API_KEY}"

datasets:
  - name: "example-dataset"
    source: "data/*.jsonl"
    input_map:
      input: "question"
      expected: "answer"
    evaluator:
      type: "exact_match"

output:
  directory: "./runs"
"""
    with open("agentbench.yaml", "w") as f:
        f.write(content)

    typer.echo("Created agentbench.yaml")

    # Create dummy agent
    if not os.path.exists("agent.py"):
        agent_content = """import sys
import json
import os

def main():
    # Read from stdin
    for line in sys.stdin:
        if not line.strip():
            continue
        data = json.loads(line)

        # Process
        input_text = data.get("input", "")

        # Output
        response = {
            "output": "I am a dummy agent. You said: " + input_text,
            "error": None
        }
        print(json.dumps(response))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
"""
        with open("agent.py", "w") as f:
            f.write(agent_content)
        typer.echo("Created agent.py")

    # Create dummy data
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/test.jsonl"):
        with open("data/test.jsonl", "w") as f:
            f.write(
                json.dumps(
                    {
                        "case_id": "1",
                        "question": "Hello",
                        "answer": "I am a dummy agent. You said: Hello",
                    }
                )
                + "\n"
            )
        typer.echo("Created data/test.jsonl")


@app.command()
def analyze(run_id: str):
    """Analyze a benchmark run."""
    # Find run directory
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        typer.echo(f"Run directory not found: {run_dir}")
        raise typer.Exit(code=1)

    print_report(run_id, run_dir)


@app.command()
def retry(
    run_id: str,
    failures_only: Annotated[
        bool,
        typer.Option(
            "--failures-only", help="Retry task failures (wrong answers) as well as system errors"
        ),
    ] = False,
):
    """Retry failed cases from a previous run."""
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        typer.echo(f"Run directory not found: {run_dir}")
        raise typer.Exit(code=1)

    # Load config from the run
    config_path = run_dir / "agentbench.yaml"
    if not config_path.exists():
        typer.echo(f"Config not found in run directory: {config_path}")
        raise typer.Exit(code=1)

    cfg = load_config(config_path)

    # Identify cases to retry
    retry_ids = set()

    # System errors
    errors_path = run_dir / "system_errors.jsonl"
    if errors_path.exists():
        with open(errors_path) as f:
            for line in f:
                try:
                    err = json.loads(line)
                    if "case_id" in err:
                        retry_ids.add(err["case_id"])
                except Exception:
                    pass

    if failures_only:
        # Load results and find passed=False
        results_path = run_dir / "results.jsonl"
        if results_path.exists():
            with open(results_path) as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        if not res.get("passed", False):
                            retry_ids.add(res["case_id"])
                    except Exception:
                        pass

    if not retry_ids:
        typer.echo("No failed cases found to retry.")
        return

    typer.echo(f"Retrying {len(retry_ids)} cases...")

    # Run
    harness = Harness(cfg, run_id=run_id)
    asyncio.run(harness.run(whitelist_ids=retry_ids))


def cli_main():
    app()


if __name__ == "__main__":
    cli_main()
