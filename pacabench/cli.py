"""PacaBench CLI - Kubernetes-style commands."""

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from pacabench.config import load_config
from pacabench.context import build_eval_context, resolve_runs_dir_from_cli
from pacabench.core import Harness
from pacabench.persistence import RunManager, get_run_summaries
from pacabench.report import (
    get_retry_case_ids,
    render_run_report,
    show_no_config,
)
from pacabench.reporters import RichProgressReporter

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)

console = Console()

_LARGE_RUN_THRESHOLD = 50


def _find_config(config_path: Path | None) -> Path | None:
    """Find the config file, checking common locations."""
    if config_path and config_path.exists():
        return config_path

    candidates = [
        Path("pacabench.yaml"),
        Path("pacabench.yml"),
        Path(".pacabench.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _resolve_run(run_id: str, runs_dir: Path):
    """Resolve a run ID (full or partial) to a RunSummary."""
    summaries = get_run_summaries(runs_dir)
    if not summaries:
        return None

    # Exact match
    for s in summaries:
        if s.run_id == run_id:
            return s

    # Partial match
    matches = [s for s in summaries if run_id in s.run_id]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        console.print(f"[yellow]Multiple runs match '{run_id}':[/yellow]")
        for m in matches[:5]:
            console.print(f"  {m.run_id}")
        return None

    return None


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
):
    """
    PacaBench - Benchmark harness for LLM agents.

    Commands:
      pacabench           Show latest run summary
      pacabench show      List all runs
      pacabench show ID   Show specific run details
      pacabench run       Start a benchmark
      pacabench retry ID  Retry failures from a run
      pacabench export ID Export run results to JSON
      pacabench init      Create new project
    """
    if ctx.invoked_subcommand is not None:
        return

    # Default behavior: show latest run
    config_file = _find_config(config)
    if not config_file:
        show_no_config()
        raise typer.Exit(0)

    try:
        cfg = load_config(config_file)
    except Exception:
        show_no_config()
        raise typer.Exit(0) from None

    runs_dir = resolve_runs_dir_from_cli(config_file, None)
    summaries = get_run_summaries(runs_dir)

    if not summaries:
        console.print()
        console.print(f"No runs yet for [bold]{cfg.name}[/bold]")
        console.print()
        console.print("Start a run with: [bold]pacabench run --limit 10[/bold]")
        console.print()
        raise typer.Exit(0)

    # Show latest completed/failed run
    actionable = [s for s in summaries if s.status in ("completed", "failed")]
    if not actionable:
        console.print()
        console.print("[yellow]No completed runs yet.[/yellow]")
        console.print("Check status with: [bold]pacabench show[/bold]")
        raise typer.Exit(0)

    latest = actionable[0]
    run_dir = runs_dir / latest.run_id
    render_run_report(run_dir, latest)


@app.command()
def show(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Run ID to show (partial match supported). Omit to list all runs."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Max items to show"),
    ] = 20,
    cases: Annotated[
        bool,
        typer.Option("--cases", help="Show individual case results"),
    ] = False,
    failures: Annotated[
        bool,
        typer.Option("--failures", "-f", help="Show only failed cases"),
    ] = False,
):
    """
    Show runs or run details.

    Without arguments: list all runs
    With run ID: show run summary
    With --cases: show individual case results
    With --failures: show only failed cases
    """
    config_file = _find_config(config)
    if not config_file:
        show_no_config()
        raise typer.Exit(1)

    runs_dir = resolve_runs_dir_from_cli(config_file, None)
    summaries = get_run_summaries(runs_dir)

    if not summaries:
        console.print("No runs found.")
        raise typer.Exit(0)

    if run_id is None:
        # List all runs
        _print_runs_table(summaries[:limit], total=len(summaries))
    else:
        # Show specific run
        run = _resolve_run(run_id, runs_dir)
        if not run:
            console.print(f"[red]Run not found: {run_id}[/red]")
            raise typer.Exit(1)

        run_dir = runs_dir / run.run_id

        if cases or failures:
            from pacabench.report import render_cases_table

            render_cases_table(run_dir, run, failures_only=failures, limit=limit)
        else:
            render_run_report(run_dir, run)


def _print_runs_table(summaries, total: int) -> None:
    """Print runs in a table format."""
    from datetime import datetime

    console.print()

    table = Table(box=None, padding=(0, 2))
    table.add_column("Run ID", style="bold", ratio=3)
    table.add_column("Status", ratio=1)
    table.add_column("Progress", justify="right", ratio=1)
    table.add_column("Cost", justify="right", ratio=1)
    table.add_column("Age", ratio=1)
    table.add_column("Agents", style="dim", ratio=2)

    for s in summaries:
        status_style = "green" if s.status == "completed" else "yellow"
        if s.status == "failed":
            status_style = "red"

        age = "-"
        if s.start_time:
            try:
                dt = datetime.fromisoformat(s.start_time)
                delta = datetime.now() - dt
                if delta.days > 0:
                    age = f"{delta.days}d"
                elif delta.seconds > 3600:
                    age = f"{delta.seconds // 3600}h"
                elif delta.seconds > 60:
                    age = f"{delta.seconds // 60}m"
                else:
                    age = "<1m"
            except Exception:
                pass

        progress = (
            f"{s.completed_cases}/{s.total_cases}" if s.total_cases else str(s.completed_cases)
        )
        cost = f"${s.total_cost_usd:.2f}" if s.total_cost_usd else "-"
        agents = ", ".join(s.agents[:2]) if s.agents else "-"
        if s.agents and len(s.agents) > 2:
            agents += f" +{len(s.agents) - 2}"

        table.add_row(
            s.run_id,
            f"[{status_style}]{s.status}[/{status_style}]",
            progress,
            cost,
            age,
            agents,
        )

    console.print(table)

    if len(summaries) < total:
        console.print(
            f"\n[dim]Showing {len(summaries)} of {total} runs. Use --limit to see more.[/dim]"
        )
    console.print()


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = Path("pacabench.yaml"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Limit number of cases per dataset"),
    ] = None,
    agents: Annotated[
        str | None,
        typer.Option("--agents", "-a", help="Comma-separated list of agents to run"),
    ] = None,
    fresh: Annotated[
        bool,
        typer.Option("--fresh", help="Force a new run (don't resume incomplete)"),
    ] = False,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Use or create a specific run ID"),
    ] = None,
    runs_dir: Annotated[
        Path | None,
        typer.Option("--runs-dir", help="Override the runs directory"),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts"),
    ] = False,
):
    """
    Start a benchmark run.

    Use --limit for quick test runs (e.g. --limit 10).
    """
    config_file = _find_config(config)
    if not config_file:
        console.print("[red]No config file found.[/red] Run 'pacabench init' first.")
        raise typer.Exit(code=1)

    agents_filter = [a.strip() for a in agents.split(",")] if agents else None

    _execute_run(
        config_path=config_file,
        limit=limit,
        agents_filter=agents_filter,
        fresh_run=fresh,
        run_id=run_id,
        runs_dir=runs_dir,
        skip_confirm=yes,
    )


@app.command()
def retry(
    run_id: Annotated[str, typer.Argument(help="Run ID to retry (partial match supported)")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
    all_failures: Annotated[
        bool,
        typer.Option("--all", help="Retry all failures, not just system errors"),
    ] = False,
):
    """
    Retry failed cases from a previous run.

    By default only retries system errors (timeouts, crashes).
    Use --all to also retry task failures (wrong answers).
    """
    config_file = _find_config(config)
    if not config_file:
        console.print("[red]No config file found.[/red]")
        raise typer.Exit(1)

    runs_dir = resolve_runs_dir_from_cli(config_file, None)
    run = _resolve_run(run_id, runs_dir)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    run_dir = runs_dir / run.run_id
    retry_ids = get_retry_case_ids(run_dir, include_task_failures=all_failures)

    if not retry_ids:
        console.print("[green]No failures to retry.[/green]")
        raise typer.Exit(0)

    console.print(f"Retrying {len(retry_ids)} cases from {run.run_id}...")
    _execute_retry(run.run_id, retry_ids, runs_dir)


@app.command()
def export(
    run_id: Annotated[str, typer.Argument(help="Run ID to export (partial match supported)")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path (default: stdout)"),
    ] = None,
):
    """
    Export run results to JSON.

    Outputs to stdout by default, use -o to write to file.
    """
    config_file = _find_config(config)
    if not config_file:
        console.print("[red]No config file found.[/red]", err=True)
        raise typer.Exit(1)

    runs_dir = resolve_runs_dir_from_cli(config_file, None)
    run = _resolve_run(run_id, runs_dir)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]", err=True)
        raise typer.Exit(1)

    run_dir = runs_dir / run.run_id
    from pacabench.report import export_run_results

    export_data = export_run_results(run_dir, run)

    if output:
        with open(output, "w") as f:
            json.dump(export_data, f, indent=2)
        console.print(f"[green]Exported to {output}[/green]", err=True)
    else:
        print(json.dumps(export_data, indent=2))


@app.command()
def init():
    """
    Initialize a new PacaBench project.

    Creates pacabench.yaml and example files.
    """
    if os.path.exists("pacabench.yaml"):
        console.print("pacabench.yaml already exists.")
        return

    content = """name: my-benchmark
description: My benchmark project
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
    with open("pacabench.yaml", "w") as f:
        f.write(content)
    console.print("Created pacabench.yaml")

    if not os.path.exists("agent.py"):
        agent_content = """import sys
import json

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        data = json.loads(line)
        input_text = data.get("input", "")

        response = {
            "output": "Echo: " + input_text,
            "error": None
        }
        print(json.dumps(response))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
"""
        with open("agent.py", "w") as f:
            f.write(agent_content)
        console.print("Created agent.py")

    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/test.jsonl"):
        with open("data/test.jsonl", "w") as f:
            f.write(
                json.dumps(
                    {
                        "case_id": "1",
                        "question": "Hello",
                        "answer": "Echo: Hello",
                    }
                )
                + "\n"
            )
        console.print("Created data/test.jsonl")


def _execute_run(
    config_path: Path,
    limit: int | None = None,
    agents_filter: list[str] | None = None,
    fresh_run: bool = False,
    run_id: str | None = None,
    runs_dir: Path | None = None,
    skip_confirm: bool = False,
):
    """Execute a benchmark run."""
    base_cfg = load_config(config_path)
    runtime_cfg = base_cfg.model_copy(deep=True)
    overrides: dict = {}

    if agents_filter:
        runtime_cfg.agents = [a for a in runtime_cfg.agents if a.name in agents_filter]
        if not runtime_cfg.agents:
            console.print(f"[red]No agents found matching: {agents_filter}[/red]")
            raise typer.Exit(code=1)
        overrides["agents"] = agents_filter

    # Warn for large runs
    estimated = len(runtime_cfg.agents) * len(runtime_cfg.datasets) * 100
    if limit:
        estimated = min(estimated, len(runtime_cfg.agents) * len(runtime_cfg.datasets) * limit)

    if not skip_confirm and not limit and estimated > _LARGE_RUN_THRESHOLD:
        console.print(
            f"\n[yellow]This will run ~{estimated} cases. Use --limit for a quick test.[/yellow]\n"
        )
        if not typer.confirm("Continue?", default=False):
            raise typer.Exit(0)

    ctx = build_eval_context(
        config_path=config_path,
        base_config=base_cfg,
        runtime_config=runtime_cfg,
        runs_dir_override=runs_dir,
        overrides=overrides,
    )

    if not run_id and not fresh_run:
        rm = RunManager(ctx)
        incomplete = rm._find_incomplete_run()
        if incomplete:
            if skip_confirm or typer.confirm(
                f"Resume incomplete run '{incomplete.name}'?", default=True
            ):
                run_id = incomplete.name
            else:
                fresh_run = True

    try:
        reporter = RichProgressReporter()
        harness = Harness(ctx, run_id=run_id, force_new_run=fresh_run, reporter=reporter)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    asyncio.run(harness.run(limit=limit))


def _execute_retry(run_id: str, retry_ids: set[str], runs_dir: Path):
    """Execute a retry for specific case IDs."""
    run_dir = runs_dir / run_id
    run_config_path = run_dir / "pacabench.yaml"

    if not run_config_path.exists():
        console.print(f"[red]Config not found in run: {run_config_path}[/red]")
        raise typer.Exit(code=1)

    cfg = load_config(run_config_path)
    runtime_cfg = cfg.model_copy(deep=True)

    ctx = build_eval_context(
        config_path=run_config_path,
        base_config=cfg,
        runtime_config=runtime_cfg,
        runs_dir_override=run_dir.parent,
    )

    reporter = RichProgressReporter()
    harness = Harness(ctx, run_id=run_id, reporter=reporter)
    asyncio.run(harness.run(whitelist_ids=retry_ids))


def cli_main():
    app()


if __name__ == "__main__":
    cli_main()
