import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agentbench.types import AggregatedMetrics, CaseResult


def load_results(run_dir: Path) -> list[CaseResult]:
    results = []
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return results

    with open(results_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                results.append(CaseResult(**data))
            except Exception:
                pass
    return results


def calculate_metrics(results: list[CaseResult]) -> AggregatedMetrics:
    total = len(results)
    if total == 0:
        return AggregatedMetrics()

    passed = sum(1 for r in results if r.passed)
    accuracy = passed / total

    # Durations
    durations = sorted([r.runner_duration_ms for r in results])
    p50_dur = durations[int(total * 0.5)] if total > 0 else 0
    p95_dur = durations[int(total * 0.95)] if total > 0 else 0

    # LLM Latency
    # r.llm_metrics is a dict. We need "llm_latency_ms".
    # It might be a list of latencies (one per call) or single.
    # Let's assume we want average latency per case? Or per call?
    # Spec: "avg_llm_latency_ms"

    latencies = []
    total_cost = 0.0
    total_input = 0
    total_output = 0

    for r in results:
        metrics = r.llm_metrics
        # Cost
        total_cost += metrics.get("llm_total_cost_usd", 0.0)
        total_input += metrics.get("llm_input_tokens", 0)
        total_output += metrics.get("llm_output_tokens", 0)

        lats = metrics.get("llm_latency_ms", [])
        if isinstance(lats, list):
            latencies.extend(lats)
        elif isinstance(lats, (int, float)):
            latencies.append(lats)

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p50_lat = sorted_lat[int(len(sorted_lat) * 0.5)] if sorted_lat else 0.0
    p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0.0

    return AggregatedMetrics(
        accuracy=accuracy,
        total_cases=total,
        failed_cases=total - passed,
        p50_duration_ms=p50_dur,
        p95_duration_ms=p95_dur,
        avg_llm_latency_ms=avg_lat,
        p50_llm_latency_ms=p50_lat,
        p95_llm_latency_ms=p95_lat,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=total_cost,
    )


def print_report(run_id: str, run_dir: Path):
    console = Console()
    results = load_results(run_dir)

    if not results:
        console.print(f"[red]No results found for run {run_id}[/red]")
        return

    # Group by Agent / Dataset
    grouped = {}
    agents = set()
    datasets = set()

    for r in results:
        key = (r.agent_name, r.dataset_name)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
        agents.add(r.agent_name)
        datasets.add(r.dataset_name)

    # Leaderboard Table
    table = Table(title=f"Leaderboard - Run {run_id}")
    table.add_column("Agent", style="cyan")
    for ds in sorted(datasets):
        table.add_column(ds, justify="right")
    table.add_column("Total Cost", justify="right")

    for agent in sorted(agents):
        row = [agent]
        agent_cost = 0.0
        for ds in sorted(datasets):
            res = grouped.get((agent, ds), [])
            metrics = calculate_metrics(res)
            agent_cost += metrics.total_cost_usd

            score = f"{metrics.accuracy:.1%}"
            if metrics.total_cases > 0:
                row.append(score)
            else:
                row.append("-")

        row.append(f"${agent_cost:.4f}")
        table.add_row(*row)

    console.print(table)

    # Detailed Metrics per Agent/Dataset
    console.print("\n[bold]Detailed Metrics[/bold]")

    for agent in sorted(agents):
        for ds in sorted(datasets):
            res = grouped.get((agent, ds), [])
            if not res:
                continue

            m = calculate_metrics(res)

            console.print(f"\n[bold blue]{agent}[/bold blue] on [bold green]{ds}[/bold green]")
            console.print(
                f"  Cases: {m.total_cases} (Passed: {m.total_cases - m.failed_cases}, Failed: {m.failed_cases})"
            )
            console.print(f"  Accuracy: {m.accuracy:.1%}")
            console.print(
                f"  Duration: p50={m.p50_duration_ms:.0f}ms, p95={m.p95_duration_ms:.0f}ms"
            )
            console.print(
                f"  LLM Latency: avg={m.avg_llm_latency_ms:.0f}ms, p50={m.p50_llm_latency_ms:.0f}ms, p95={m.p95_llm_latency_ms:.0f}ms"
            )
            console.print(f"  Tokens: In={m.total_input_tokens}, Out={m.total_output_tokens}")
            console.print(f"  Cost: ${m.total_cost_usd:.8f}")

    # System Errors
    errors_path = run_dir / "system_errors.jsonl"
    if errors_path.exists():
        console.print("\n[bold red]System Errors[/bold red]")
        with open(errors_path) as f:
            lines = f.readlines()
            console.print(f"  Total Errors: {len(lines)}")
            # Print top 5
            for line in lines[:5]:
                try:
                    err = json.loads(line)
                    console.print(f"  Case {err.get('case_id')}: {err.get('error')}")
                except Exception:
                    console.print(f"  {line.strip()}")
