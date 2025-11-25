"""Report module for displaying run results."""

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from pacabench.analysis import (
    calculate_metrics,
    calculate_metrics_fast,
    load_results,
    load_results_raw,
)
from pacabench.persistence import RunSummary
from pacabench.types import CaseResult


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


def render_run_report(run_dir: Path, summary: RunSummary) -> None:
    """Render a comprehensive run report to the console."""
    console = Console()
    results = load_results_raw(run_dir)  # Fast: raw dicts
    system_errors = _load_system_errors(run_dir)

    # Header
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

    # Group by agent (using raw dicts)
    grouped: dict[str, list[dict]] = {}
    for r in results:
        agent = r.get("agent_name", "unknown")
        if agent not in grouped:
            grouped[agent] = []
        grouped[agent].append(r)

    # Per-agent detailed report
    for agent_name in sorted(grouped.keys()):
        agent_results = grouped[agent_name]
        m = calculate_metrics_fast(agent_results)  # Fast: no Pydantic

        total = m.total_cases
        passed = total - m.failed_cases
        accuracy = m.accuracy
        score_color = "green" if accuracy >= 0.8 else "yellow" if accuracy >= 0.5 else "red"

        console.print(f"[bold cyan]{agent_name}[/bold cyan]")
        console.print()

        # Score row
        bar = _progress_bar(accuracy, width=20)
        console.print(
            f"  Score      [{score_color}]{accuracy:>6.1%}[/{score_color}]  {bar}  {passed}/{total} passed"
        )

        # Latency row
        console.print(
            f"  Latency    [dim]p50[/dim] {_format_duration(m.p50_duration_ms):>6}  "
            f"[dim]p95[/dim] {_format_duration(m.p95_duration_ms):>6}  "
            f"[dim]avg[/dim] {_format_duration(m.avg_llm_latency_ms):>6}"
        )

        # Tokens row
        total_tokens = m.total_input_tokens + m.total_output_tokens
        console.print(
            f"  Tokens     [dim]in[/dim] {_format_tokens(m.total_input_tokens):>6}  "
            f"[dim]out[/dim] {_format_tokens(m.total_output_tokens):>6}  "
            f"[dim]total[/dim] {_format_tokens(total_tokens):>6}"
        )

        # Cost row
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

    # Failures summary (using raw dicts)
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


def export_run_results(run_dir: Path, summary: RunSummary) -> dict:
    """Export run results as a dictionary for JSON output."""
    results = load_results(run_dir)
    system_errors = _load_system_errors(run_dir)

    grouped: dict[str, list[CaseResult]] = {}
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


def get_retry_case_ids(run_dir: Path, include_task_failures: bool = False) -> set[str]:
    """Get case IDs that should be retried."""
    retry_ids: set[str] = set()

    # System errors (always included)
    errors = _load_system_errors(run_dir)
    for e in errors:
        if e.get("case_id"):
            retry_ids.add(e["case_id"])

    # Task failures (optional)
    if include_task_failures:
        results = load_results(run_dir)
        for r in results:
            if not r.passed:
                retry_ids.add(r.case_id)

    return retry_ids


def render_cases_table(
    run_dir: Path, summary: RunSummary, failures_only: bool = False, limit: int = 50
) -> None:
    """Render a table of individual case results."""
    console = Console()
    results = load_results(run_dir)
    system_errors = _load_system_errors(run_dir)

    # Build system error lookup
    sys_error_ids = {e.get("case_id") for e in system_errors}

    # Filter if needed
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
        # Determine status
        if r.case_id in sys_error_ids:
            status = "[red]error[/red]"
        elif r.passed:
            status = "[green]pass[/green]"
        else:
            status = "[red]fail[/red]"

        # Truncate output
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


def show_no_config() -> None:
    """Show message when no config file is found."""
    console = Console()
    console.print()
    console.print("[yellow]No pacabench.yaml found.[/yellow]")
    console.print("Run [bold]pacabench init[/bold] to get started.")
    console.print()
