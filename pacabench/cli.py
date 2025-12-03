"""PacaBench CLI - Kubernetes-style commands."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pacabench.models import load_config
from pacabench.storage import (
    calculate_metrics,
    get_run_summaries,
    load_results,
    load_results_raw,
)

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


def _resolve_runs_dir(config_path: Path | None, override: Path | None = None) -> Path:
    """Resolve the runs directory from config or defaults."""
    if override:
        return override.expanduser()

    if config_path and config_path.exists():
        try:
            cfg = load_config(config_path)
            runs_dir = Path(cfg.output.directory).expanduser()
            if not runs_dir.is_absolute():
                runs_dir = (config_path.parent / runs_dir).resolve()
            runs_dir.mkdir(parents=True, exist_ok=True)
            return runs_dir
        except Exception:
            pass

    env_dir = os.getenv("PACABENCH_RUNS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.cwd() / "runs"


def _resolve_run(run_id: str, runs_dir: Path):
    """Resolve a run ID (full or partial) to a RunSummary."""
    summaries = get_run_summaries(runs_dir)
    if not summaries:
        return None

    for s in summaries:
        if s.run_id == run_id:
            return s

    matches = [s for s in summaries if run_id in s.run_id]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        console.print(f"[yellow]Multiple runs match '{run_id}':[/yellow]")
        for m in matches[:5]:
            console.print(f"  {m.run_id}")
        return None

    return None


def _show_no_config() -> None:
    """Show message when no config file is found."""
    console.print()
    console.print("[yellow]No pacabench.yaml found.[/yellow]")
    console.print("Run [bold]pacabench init[/bold] to get started.")
    console.print()


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

    config_file = _find_config(config)
    if not config_file:
        _show_no_config()
        raise typer.Exit(0)

    try:
        cfg = load_config(config_file)
    except Exception:
        _show_no_config()
        raise typer.Exit(0) from None

    runs_dir = _resolve_runs_dir(config_file, None)
    summaries = get_run_summaries(runs_dir)

    if not summaries:
        console.print()
        console.print(f"No runs yet for [bold]{cfg.name}[/bold]")
        console.print()
        console.print("Start a run with: [bold]pacabench run --limit 10[/bold]")
        console.print()
        raise typer.Exit(0)

    actionable = [s for s in summaries if s.status in ("completed", "failed")]
    if not actionable:
        console.print()
        console.print("[yellow]No completed runs yet.[/yellow]")
        console.print("Check status with: [bold]pacabench show[/bold]")
        raise typer.Exit(0)

    latest = actionable[0]
    run_dir = runs_dir / latest.run_id
    _render_run_report(run_dir, latest)


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
        _show_no_config()
        raise typer.Exit(1)

    runs_dir = _resolve_runs_dir(config_file, None)
    summaries = get_run_summaries(runs_dir)

    if not summaries:
        console.print("No runs found.")
        raise typer.Exit(0)

    if run_id is None:
        _print_runs_table(summaries[:limit], total=len(summaries))
    else:
        run = _resolve_run(run_id, runs_dir)
        if not run:
            console.print(f"[red]Run not found: {run_id}[/red]")
            raise typer.Exit(1)

        run_dir = runs_dir / run.run_id

        if cases or failures:
            _render_cases_table(run_dir, run, failures_only=failures, limit=limit)
        else:
            _render_run_report(run_dir, run)


def _print_runs_table(summaries, total: int) -> None:
    """Print runs in a table format."""
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
        runs_dir_override=runs_dir,
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

    runs_dir = _resolve_runs_dir(config_file, None)
    run = _resolve_run(run_id, runs_dir)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    run_dir = runs_dir / run.run_id
    retry_ids = _get_retry_case_ids(run_dir, include_task_failures=all_failures)

    if not retry_ids:
        console.print("[green]No failures to retry.[/green]")
        raise typer.Exit(0)

    console.print(f"Retrying {len(retry_ids)} cases from {run.run_id}...")
    _execute_retry(run.run_id, retry_ids, runs_dir)


@app.command("export")
def export_cmd(
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

    runs_dir = _resolve_runs_dir(config_file, None)
    run = _resolve_run(run_id, runs_dir)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]", err=True)
        raise typer.Exit(1)

    run_dir = runs_dir / run.run_id
    export_data = _export_run_results(run_dir, run)

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


# --- Internal helpers ---


def _execute_run(
    config_path: Path,
    limit: int | None = None,
    agents_filter: list[str] | None = None,
    fresh_run: bool = False,
    run_id: str | None = None,
    runs_dir_override: Path | None = None,
    skip_confirm: bool = False,
):
    """Execute a benchmark run."""
    import asyncio

    from pacabench.engine import Harness, RichProgressReporter
    from pacabench.storage import RunManager

    base_cfg = load_config(config_path)
    runtime_cfg = base_cfg.model_copy(deep=True)
    overrides: dict = {}

    if agents_filter:
        runtime_cfg.agents = [a for a in runtime_cfg.agents if a.name in agents_filter]
        if not runtime_cfg.agents:
            console.print(f"[red]No agents found matching: {agents_filter}[/red]")
            raise typer.Exit(code=1)
        overrides["agents"] = agents_filter

    estimated = len(runtime_cfg.agents) * len(runtime_cfg.datasets) * 100
    if limit:
        estimated = min(estimated, len(runtime_cfg.agents) * len(runtime_cfg.datasets) * limit)

    if not skip_confirm and not limit and estimated > _LARGE_RUN_THRESHOLD:
        console.print(
            f"\n[yellow]This will run ~{estimated} cases. Use --limit for a quick test.[/yellow]\n"
        )
        if not typer.confirm("Continue?", default=False):
            raise typer.Exit(0)

    runs_dir = _resolve_runs_dir(config_path, runs_dir_override)
    env = os.environ.copy()

    if not run_id and not fresh_run:
        rm = RunManager(
            config=runtime_cfg,
            base_config=base_cfg,
            runs_dir=runs_dir,
            config_path=config_path,
            overrides=overrides,
        )
        incomplete = rm.find_incomplete_run()
        if incomplete:
            if skip_confirm or typer.confirm(
                f"Resume incomplete run '{incomplete.name}'?", default=True
            ):
                run_id = incomplete.name
            else:
                fresh_run = True

    try:
        reporter = RichProgressReporter()
        harness = Harness(
            config=runtime_cfg,
            base_config=base_cfg,
            runs_dir=runs_dir,
            config_path=config_path,
            env=env,
            overrides=overrides,
            run_id=run_id,
            force_new_run=fresh_run,
            reporter=reporter,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    asyncio.run(harness.run(limit=limit))


def _execute_retry(run_id: str, retry_ids: set[str], runs_dir: Path):
    """Execute a retry for specific case IDs."""
    import asyncio

    from pacabench.engine import Harness, RichProgressReporter

    run_dir = runs_dir / run_id
    run_config_path = run_dir / "pacabench.yaml"

    if not run_config_path.exists():
        console.print(f"[red]Config not found in run: {run_config_path}[/red]")
        raise typer.Exit(code=1)

    cfg = load_config(run_config_path)
    runtime_cfg = cfg.model_copy(deep=True)
    env = os.environ.copy()

    reporter = RichProgressReporter()
    harness = Harness(
        config=runtime_cfg,
        base_config=cfg,
        runs_dir=run_dir.parent,
        config_path=run_config_path,
        env=env,
        run_id=run_id,
        reporter=reporter,
    )
    asyncio.run(harness.run(whitelist_ids=retry_ids))


def _get_retry_case_ids(run_dir: Path, include_task_failures: bool = False) -> set[str]:
    """Get case IDs that should be retried."""
    retry_ids: set[str] = set()

    errors_path = run_dir / "system_errors.jsonl"
    if errors_path.exists():
        with open(errors_path) as f:
            for line in f:
                try:
                    err = json.loads(line)
                    if err.get("case_id"):
                        retry_ids.add(err["case_id"])
                except json.JSONDecodeError:
                    continue

    if include_task_failures:
        results = load_results(run_dir)
        for r in results:
            if not r.passed:
                retry_ids.add(r.case_id)

    return retry_ids


# --- Report rendering ---


def _time_ago(dt: datetime) -> str:
    """Format a datetime as a human-readable 'time ago' string."""
    now = datetime.now()
    delta = now - dt
    seconds = delta.total_seconds()

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins}m ago"
    if seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    days = int(seconds / 86400)
    return f"{days}d ago"


def _progress_bar(ratio: float, width: int = 10) -> str:
    """Create a simple progress bar string."""
    filled = int(ratio * width)
    empty = width - filled
    return "█" * filled + "·" * empty


def _format_duration(ms: float) -> str:
    """Format milliseconds to human readable."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms / 1000:.1f}s"
    else:
        return f"{ms / 60000:.1f}m"


def _format_tokens(tokens: int) -> str:
    """Format token count."""
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1000000:
        return f"{tokens / 1000:.1f}k"
    else:
        return f"{tokens / 1000000:.2f}M"


def _load_system_errors(run_dir: Path) -> list[dict]:
    """Load system errors from a run directory."""
    errors_path = run_dir / "system_errors.jsonl"
    errors = []
    if errors_path.exists():
        with open(errors_path) as f:
            for line in f:
                try:
                    errors.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return errors


def _render_run_report(run_dir: Path, summary) -> None:
    """Render a comprehensive run report to the console."""
    results = load_results_raw(run_dir)
    system_errors = _load_system_errors(run_dir)

    console.print()
    status_color = "green" if summary.status == "completed" else "yellow"
    if summary.status == "failed":
        status_color = "red"

    time_str = ""
    if summary.start_time:
        try:
            dt = datetime.fromisoformat(summary.start_time)
            time_str = _time_ago(dt)
        except Exception:
            pass

    header = Text()
    header.append(summary.run_id, style="bold")
    header.append("  ")
    header.append(f"{summary.status} {time_str}", style=status_color)
    console.print(header)
    console.print()

    if not results:
        console.print("[dim]No results yet.[/dim]")
        console.print()
        return

    grouped: dict[str, list[dict]] = {}
    for r in results:
        agent = r.get("agent_name", "unknown")
        if agent not in grouped:
            grouped[agent] = []
        grouped[agent].append(r)

    for agent_name in sorted(grouped.keys()):
        agent_results = grouped[agent_name]
        m = calculate_metrics(agent_results)

        total = m.total_cases
        passed = total - m.failed_cases
        accuracy = m.accuracy
        score_color = "green" if accuracy >= 0.8 else "yellow" if accuracy >= 0.5 else "red"

        console.print(f"[bold cyan]{agent_name}[/bold cyan]")
        console.print()

        bar = _progress_bar(accuracy, width=20)
        console.print(
            f"  Score      [{score_color}]{accuracy:>6.1%}[/{score_color}]  {bar}  {passed}/{total} passed"
        )

        console.print(
            f"  Latency    [dim]p50[/dim] {_format_duration(m.p50_duration_ms):>6}  "
            f"[dim]p95[/dim] {_format_duration(m.p95_duration_ms):>6}  "
            f"[dim]avg[/dim] {_format_duration(m.avg_llm_latency_ms):>6}"
        )

        total_tokens = m.total_input_tokens + m.total_output_tokens
        console.print(
            f"  Tokens     [dim]in[/dim] {_format_tokens(m.total_input_tokens):>6}  "
            f"[dim]out[/dim] {_format_tokens(m.total_output_tokens):>6}  "
            f"[dim]total[/dim] {_format_tokens(total_tokens):>6}"
        )

        total_cost = m.total_cost_usd + m.total_judge_cost_usd
        if m.total_judge_cost_usd > 0:
            console.print(
                f"  Cost       [dim]agent[/dim] ${m.total_cost_usd:>5.3f}  "
                f"[dim]judge[/dim] ${m.total_judge_cost_usd:>5.3f}  "
                f"[bold]total[/bold] ${total_cost:.3f}"
            )
        else:
            console.print(f"  Cost       ${total_cost:.4f}")

        console.print()

    task_failures = [
        r for r in results if not r.get("passed") and r.get("error_type") != "system_failure"
    ]
    sys_errors = len(system_errors)

    if task_failures or sys_errors:
        console.print("[bold]FAILURES[/bold]")
        if task_failures:
            console.print(f"  [red]{len(task_failures)} task failures[/red] (wrong answers)")
        if sys_errors:
            console.print(f"  [yellow]{sys_errors} system errors[/yellow] (crashes/timeouts)")
        console.print()
        console.print("[dim]View with: pacabench show <run> --failures[/dim]")
        console.print("[dim]Retry with: pacabench retry <run>[/dim]")
    else:
        console.print("[green]All cases passed![/green]")

    console.print()


def _render_cases_table(
    run_dir: Path, summary, failures_only: bool = False, limit: int = 50
) -> None:
    """Render a table of individual case results."""
    results = load_results(run_dir)
    system_errors = _load_system_errors(run_dir)

    sys_error_ids = {e.get("case_id") for e in system_errors}

    if failures_only:
        results = [r for r in results if not r.passed or r.case_id in sys_error_ids]

    console.print()
    console.print(f"[bold]{summary.run_id}[/bold]")

    if not results:
        console.print()
        if failures_only:
            console.print("[green]No failures.[/green]")
        else:
            console.print("[dim]No cases.[/dim]")
        console.print()
        return

    total = len(results)
    results = results[:limit]

    console.print()

    table = Table(box=None, padding=(0, 2))
    table.add_column("Case ID", style="dim", ratio=2)
    table.add_column("Agent", ratio=2)
    table.add_column("Status", ratio=1)
    table.add_column("Output", ratio=4)
    table.add_column("Duration", justify="right", ratio=1)

    for r in results:
        if r.case_id in sys_error_ids:
            status = "[red]error[/red]"
        elif r.passed:
            status = "[green]pass[/green]"
        else:
            status = "[red]fail[/red]"

        output = r.output or r.error or "-"
        if len(output) > 60:
            output = output[:57] + "..."

        duration = f"{r.runner_duration_ms:.0f}ms" if r.runner_duration_ms else "-"

        table.add_row(
            r.case_id,
            r.agent_name,
            status,
            output,
            duration,
        )

    console.print(table)

    if len(results) < total:
        console.print(f"\n[dim]Showing {len(results)} of {total}. Use --limit to see more.[/dim]")
    console.print()


def _export_run_results(run_dir: Path, summary) -> dict:
    """Export run results as a dictionary for JSON output."""
    results = load_results(run_dir)
    system_errors = _load_system_errors(run_dir)

    grouped: dict[str, list] = {}
    for r in results:
        if r.agent_name not in grouped:
            grouped[r.agent_name] = []
        grouped[r.agent_name].append(r)

    export_data = {
        "run_id": summary.run_id,
        "status": summary.status,
        "start_time": summary.start_time,
        "total_cases": summary.total_cases,
        "completed_cases": summary.completed_cases,
        "agents": {},
        "system_errors": system_errors,
    }

    for agent_name in sorted(grouped.keys()):
        agent_results = grouped[agent_name]
        m = calculate_metrics(agent_results)
        export_data["agents"][agent_name] = {
            "metrics": m.model_dump(),
            "results": [
                {
                    "case_id": r.case_id,
                    "passed": r.passed,
                    "output": r.output,
                    "error": r.error,
                }
                for r in agent_results
            ],
        }

    return export_data


def cli_main():
    app()


if __name__ == "__main__":
    cli_main()
