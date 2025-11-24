import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.console import Console

from pacabench.dashboard import DashboardRenderer, DashboardState
from pacabench.types import AggregatedMetrics, CaseResult


def load_results(run_dir: Path) -> list[CaseResult]:
    results = []
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return results

    # Dedup results: keep only the LAST result for each (agent, dataset, case_id)
    # Use 'attempt' to tiebreak if available, otherwise strict file order
    deduped = {}
    with open(results_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                res = CaseResult(**data)
                key = (res.agent_name, res.dataset_name, res.case_id)
                deduped[key] = res
            except Exception:
                pass
    return list(deduped.values())


def _calculate_percentile(data: list[float], percentile: float) -> float:
    """Calculate percentile using linear interpolation (numpy-style)."""
    if not data:
        return 0.0
    data.sort()
    index = (len(data) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return data[int(index)]
    return data[lower] * (upper - index) + data[upper] * (index - lower)


def calculate_metrics(results: list[CaseResult]) -> AggregatedMetrics:
    total = len(results)
    if total == 0:
        return AggregatedMetrics()

    passed = sum(1 for r in results if r.passed)
    accuracy = passed / total

    # Durations
    durations = [r.runner_duration_ms for r in results]
    p50_dur = _calculate_percentile(durations, 0.50)
    p95_dur = _calculate_percentile(durations, 0.95)

    # LLM Latency
    latencies = []
    total_cost = 0.0
    total_judge_cost = 0.0
    total_input = 0
    total_output = 0
    judge_total_input = 0
    judge_total_output = 0

    for r in results:
        metrics = r.llm_metrics
        # Cost
        total_cost += metrics.get("llm_total_cost_usd", 0.0)
        total_input += metrics.get("llm_input_tokens", 0)
        total_output += metrics.get("llm_output_tokens", 0)

        # Judge Metrics
        total_judge_cost += r.judge_cost_usd or 0.0
        if r.judge_metrics:
            judge_total_input += r.judge_metrics.get("input_tokens", 0)
            judge_total_output += r.judge_metrics.get("output_tokens", 0)
            # Assume judge latency is not explicitly tracked in metrics yet unless added
            # If judge metrics had latency, we'd add it here.

        lats = metrics.get("llm_latency_ms", [])
        if isinstance(lats, list):
            latencies.extend(lats)
        elif isinstance(lats, (int, float)):
            latencies.append(lats)

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p50_lat = _calculate_percentile(latencies, 0.50)
    p95_lat = _calculate_percentile(latencies, 0.95)

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
        total_judge_cost_usd=total_judge_cost,
    )


def print_report(
    run_id: str, run_dir: Path, output_format: Literal["text", "json", "markdown"] = "text"
):
    results = load_results(run_dir)
    if not results:
        if output_format == "text":
            Console().print(f"[red]No results found for run {run_id}[/red]")
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

    # Load system errors
    errors_path = run_dir / "system_errors.jsonl"
    system_errors = []
    if errors_path.exists():
        with open(errors_path) as f:
            for line in f:
                try:
                    system_errors.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Load metadata for timing
    metadata_path = run_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except Exception:
            pass

    if output_format == "json":
        _print_json_report(run_id, agents, datasets, grouped, system_errors)
    elif output_format == "markdown":
        _print_markdown_report(run_id, agents, datasets, grouped, system_errors)
    else:
        _print_text_report(run_id, agents, datasets, grouped, system_errors, metadata)


def _print_json_report(run_id, agents, datasets, grouped, system_errors):
    report = {"run_id": run_id, "leaderboard": [], "detailed": {}, "system_errors": system_errors}

    for agent in sorted(agents):
        agent_data = {"agent": agent, "total_cost": 0.0, "scores": {}}
        for ds in sorted(datasets):
            res = grouped.get((agent, ds), [])
            if not res:
                continue
            m = calculate_metrics(res)
            agent_data["total_cost"] += m.total_cost_usd + m.total_judge_cost_usd
            agent_data["scores"][ds] = {
                "accuracy": m.accuracy,
                "total_cases": m.total_cases,
                "metrics": m.model_dump(),
            }

            if agent not in report["detailed"]:
                report["detailed"][agent] = {}
            report["detailed"][agent][ds] = m.model_dump()

        report["leaderboard"].append(agent_data)

    print(json.dumps(report, indent=2))


def _print_markdown_report(run_id, agents, datasets, grouped, system_errors):
    print(f"# PacaBench Run Report: {run_id}\n")

    print("## Leaderboard\n")
    headers = ["Agent"] + sorted(datasets) + ["Total Cost"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for agent in sorted(agents):
        row = [agent]
        total_cost = 0.0
        for ds in sorted(datasets):
            res = grouped.get((agent, ds), [])
            if res:
                m = calculate_metrics(res)
                total_cost += m.total_cost_usd + m.total_judge_cost_usd
                row.append(f"{m.accuracy:.1%}")
            else:
                row.append("-")
        row.append(f"${total_cost:.4f}")
        print("| " + " | ".join(row) + " |")

    print("\n## Detailed Metrics\n")
    for agent in sorted(agents):
        print(f"### Agent: {agent}")
        for ds in sorted(datasets):
            res = grouped.get((agent, ds), [])
            if not res:
                continue
            m = calculate_metrics(res)
            print(f"#### Dataset: {ds}")
            print(
                f"- **Accuracy**: {m.accuracy:.1%} ({m.total_cases - m.failed_cases}/{m.total_cases})"
            )
            print(f"- **Duration**: p50={m.p50_duration_ms:.0f}ms, p95={m.p95_duration_ms:.0f}ms")
            print(
                f"- **LLM Latency**: avg={m.avg_llm_latency_ms:.0f}ms, p50={m.p50_llm_latency_ms:.0f}ms"
            )
            print(f"- **Cost**: ${m.total_cost_usd + m.total_judge_cost_usd:.6f}")
            print("")

    if system_errors:
        print("\n## System Errors\n")
        print(f"Total Errors: {len(system_errors)}\n")
        print("| Case ID | Agent | Dataset | Error |")
        print("| --- | --- | --- | --- |")
        for err in system_errors[:20]:
            print(
                f"| {err.get('case_id')} | {err.get('agent_name')} | {err.get('dataset_name')} | {err.get('error')} |"
            )
        if len(system_errors) > 20:
            print(f"\n... and {len(system_errors) - 20} more.")


def _print_text_report(run_id, agents, datasets, grouped, system_errors, metadata):
    console = Console()
    console.print(f"\n[bold]Analysis Report for Run: {run_id}[/bold]\n")

    # Build Dashboard State from historical data
    state = DashboardState()

    # Calculate duration from metadata
    start_str = metadata.get("start_time")
    end_str = metadata.get("completed_time") or metadata.get("last_resumed")
    duration_sec = 0.0
    if start_str:
        try:
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str) if end_str else datetime.now()
            duration_sec = (end_dt - start_dt).total_seconds()
        except Exception:
            pass

    # Trick dashboard renderer to show correct elapsed time
    state.start_time = time.time() - duration_sec

    # Populate states
    overall_cost = 0.0
    for agent in agents:
        for ds in datasets:
            res = grouped.get((agent, ds), [])
            if not res:
                # Could initialize with 0 if we know it was supposed to run
                continue

            # Aggregate
            m = calculate_metrics(res)

            ds_state = state.get_state(agent, ds)
            ds_state.status = (
                "Completed"  # Or 'Failed' if metadata says so? Assume Completed for report
            )
            ds_state.total_cases = m.total_cases
            ds_state.completed_cases = m.total_cases
            ds_state.passed_cases = m.total_cases - m.failed_cases
            ds_state.failed_cases = m.failed_cases
            ds_state.total_cost = m.total_cost_usd + m.total_judge_cost_usd
            ds_state.avg_latency_ms = m.avg_llm_latency_ms
            ds_state.last_case_id = res[-1].case_id if res else None

            # Count errors from system_errors list?
            # The system_errors list is global. Let's filter it.
            agent_ds_errors = [
                e
                for e in system_errors
                if e.get("agent_name") == agent and e.get("dataset_name") == ds
            ]
            ds_state.error_cases = len(agent_ds_errors)

            overall_cost += ds_state.total_cost

    state.total_cost = overall_cost
    state.circuit_open = metadata.get("status") == "failed"

    # Render Dashboard
    renderer = DashboardRenderer()
    console.print(renderer.render(state))

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
            console.print(
                f"  Cost: ${m.total_cost_usd:.6f} (Agent) + ${m.total_judge_cost_usd:.6f} (Judge) = ${m.total_cost_usd + m.total_judge_cost_usd:.6f}"
            )

    # System Errors
    if system_errors:
        console.print("\n[bold red]System Errors[/bold red]")
        console.print(f"  Total Errors: {len(system_errors)}")
        for entry in system_errors[:5]:
            case_id = entry.get("case_id")
            agent = entry.get("agent_name")
            dataset = entry.get("dataset_name")
            error = entry.get("error")
            console.print(f"  {case_id} ({agent} on {dataset}): {error}")
