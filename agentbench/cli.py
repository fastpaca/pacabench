import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from agentbench.analysis import print_report
from agentbench.config import load_config
from agentbench.context import build_eval_context, resolve_run_directory, resolve_runs_dir_from_cli
from agentbench.core import Harness
from agentbench.persistence import RunManager, find_latest_run

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
    runs_dir: Annotated[
        Path | None,
        typer.Option("--runs-dir", help="Override base runs directory (defaults to config output)"),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Use or create a specific run directory name"),
    ] = None,
    fresh_run: Annotated[
        bool,
        typer.Option("--fresh-run", help="Force a brand new run even if incomplete runs exist"),
    ] = False,
):
    """Execute a benchmark run."""
    if not config.exists():
        typer.echo(f"Config file not found: {config}")
        raise typer.Exit(code=1)

    base_cfg = load_config(config)
    runtime_cfg = base_cfg.model_copy(deep=True)
    overrides: dict[str, Any] = {}

    if concurrency:
        runtime_cfg.config.concurrency = concurrency
        overrides["concurrency"] = concurrency

    if agents:
        agent_names = [a.strip() for a in agents.split(",")]
        runtime_cfg.agents = [a for a in runtime_cfg.agents if a.name in agent_names]
        if not runtime_cfg.agents:
            typer.echo(f"No agents found matching: {agents}")
            raise typer.Exit(code=1)
        overrides["agents"] = agent_names

    ctx = build_eval_context(
        config_path=config,
        base_config=base_cfg,
        runtime_config=runtime_cfg,
        runs_dir_override=runs_dir,
        overrides=overrides,
    )

    # Check for incomplete run if not specified
    if not run_id and not fresh_run:
        # Use a temporary RunManager to check
        rm = RunManager(ctx)
        incomplete = rm._find_incomplete_run()
        if incomplete:
            should_resume = typer.confirm(
                f"Found incomplete run '{incomplete.name}'. Resume?", default=True
            )
            if should_resume:
                run_id = incomplete.name
            else:
                fresh_run = True

    try:
        harness = Harness(ctx, run_id=run_id, force_new_run=fresh_run)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
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
    base_url: "https://api.openai.com/v1"

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
def analyze(
    run_id: Annotated[
        str | None,
        typer.Argument(
            help="Run ID to analyze. If not provided, defaults to the latest completed run."
        ),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to configuration file")
    ] = Path("agentbench.yaml"),
    runs_dir: Annotated[
        Path | None,
        typer.Option("--runs-dir", help="Override base runs directory (defaults to config output)"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: text, json, or markdown"),
    ] = "text",
):
    """Analyze a benchmark run."""
    resolved_runs_dir = resolve_runs_dir_from_cli(config, runs_dir)

    if not run_id:
        latest_run_path = find_latest_run(resolved_runs_dir)
        if not latest_run_path:
            typer.echo(f"No completed runs found in {resolved_runs_dir}.")
            raise typer.Exit(code=1)
        run_id = latest_run_path.name
        typer.echo(f"Analyzing latest run: {run_id}")

    try:
        run_dir = resolve_run_directory(run_id, resolved_runs_dir)
    except FileNotFoundError:
        typer.echo(
            f"Run directory not found for id '{run_id}'. Searched under {resolved_runs_dir}."
        )
        raise typer.Exit(code=1) from None

    if output_format not in ["text", "json", "markdown"]:
        typer.echo(f"Invalid format: {output_format}. Must be text, json, or markdown.")
        raise typer.Exit(code=1)

    print_report(run_id, run_dir, output_format=output_format)


@app.command()
def retry(
    run_id: str,
    failures_only: Annotated[
        bool,
        typer.Option(
            "--failures-only", help="Retry task failures (wrong answers) as well as system errors"
        ),
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to configuration file")
    ] = Path("agentbench.yaml"),
    runs_dir: Annotated[
        Path | None,
        typer.Option("--runs-dir", help="Override base runs directory (defaults to config output)"),
    ] = None,
):
    """Retry failed cases from a previous run."""
    resolved_runs_dir = resolve_runs_dir_from_cli(config, runs_dir)
    try:
        run_dir = resolve_run_directory(run_id, resolved_runs_dir)
    except FileNotFoundError:
        typer.echo(
            f"Run directory not found for id '{run_id}'. Searched under {resolved_runs_dir}."
        )
        raise typer.Exit(code=1) from None

    # Load config from the run
    config_path = run_dir / "agentbench.yaml"
    if not config_path.exists():
        typer.echo(f"Config not found in run directory: {config_path}")
        raise typer.Exit(code=1)

    cfg = load_config(config_path)

    # Identify cases to retry
    retry_ids: set[str] = set()
    system_error_ids: set[str] = set()
    task_failure_ids: set[str] = set()

    # System errors
    errors_path = run_dir / "system_errors.jsonl"
    if errors_path.exists():
        with open(errors_path) as f:
            for line in f:
                try:
                    err = json.loads(line)
                    if "case_id" in err:
                        system_error_ids.add(err["case_id"])
                except Exception:
                    pass

    retry_ids.update(system_error_ids)

    if failures_only:
        # Load results and find passed=False
        results_path = run_dir / "results.jsonl"
        if results_path.exists():
            with open(results_path) as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        if not res.get("passed", False):
                            task_failure_ids.add(res["case_id"])
                    except Exception:
                        pass

        retry_ids.update(task_failure_ids)

    if not retry_ids:
        typer.echo("No failed cases found to retry.")
        return

    if failures_only:
        typer.echo(
            f"Retrying {len(retry_ids)} cases "
            f"(system errors: {len(system_error_ids)}, task failures: {len(task_failure_ids)})."
        )
    else:
        typer.echo(f"Retrying {len(system_error_ids)} system-error cases...")

    # Run
    base_cfg = cfg
    runtime_cfg = base_cfg.model_copy(deep=True)
    ctx = build_eval_context(
        config_path=config_path,
        base_config=base_cfg,
        runtime_config=runtime_cfg,
        runs_dir_override=run_dir.parent,
    )
    harness = Harness(ctx, run_id=run_id)
    asyncio.run(harness.run(whitelist_ids=retry_ids))


def cli_main():
    app()


if __name__ == "__main__":
    cli_main()
