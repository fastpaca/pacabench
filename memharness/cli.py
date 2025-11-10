"""CLI for memharness."""

import asyncio
import json
import traceback
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from memharness.configs import ANSWERERS, DATASETS

# Constants
_DEFAULT_CONCURRENCY = 10
_SEPARATOR = "=" * 60
_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"

app = typer.Typer(
    name="memharness",
    help="Memory QA benchmark harness for evaluating memory systems.",
    add_completion=False,
)


def _print_available_options(title: str, options: dict) -> None:
    """Print available options in a consistent format."""
    print(f"{title}:")
    for name in sorted(options.keys()):
        print(f"  {name}")


def _validate_choice(
    value: str | None,
    options: dict,
    option_type: str,
) -> str:
    """Validate and return a choice from available options.

    Args:
        value: User-provided value
        options: Dict of available options
        option_type: Type name for error messages (e.g., "dataset", "config")

    Returns:
        The validated value

    Raises:
        typer.Exit: If validation fails
    """
    if value is None:
        print(f"Error: --{option_type} is required\n")
        _print_available_options(f"Available {option_type}s", options)
        print()
        raise typer.Exit(1)

    if value not in options:
        print(f"Error: Unknown {option_type} '{value}'\n")
        _print_available_options(f"Available {option_type}s", options)
        print()
        raise typer.Exit(1)

    return value


def _calculate_percentile(sorted_values: list[float], percentile: float) -> float:
    """Calculate percentile using linear interpolation.

    Args:
        sorted_values: Pre-sorted list of values
        percentile: Percentile to calculate (0.0 to 1.0)

    Returns:
        The percentile value
    """
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return sorted_values[0]

    # Use linear interpolation between closest ranks
    index = percentile * (len(sorted_values) - 1)
    lower_idx = int(index)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    weight = index - lower_idx

    return sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight


def _compute_token_metrics(cases: list) -> dict[str, int]:
    """Compute aggregated token metrics from cases.

    Args:
        cases: List of report cases

    Returns:
        Dict with total_input, total_output token counts
    """
    total_input = sum(case.metrics.get("input_tokens", 0) for case in cases)
    total_output = sum(case.metrics.get("output_tokens", 0) for case in cases)
    return {
        "total_input": total_input,
        "total_output": total_output,
    }


def _compute_latency_metrics(cases: list) -> dict[str, float]:
    """Compute latency metrics including percentiles.

    Args:
        cases: List of report cases

    Returns:
        Dict with duration statistics
    """
    durations = [case.total_duration for case in cases]
    task_durations = [case.task_duration for case in cases]

    if not durations:
        return {
            "total": 0.0,
            "avg_task": 0.0,
            "avg_total": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    sorted_durations = sorted(durations)
    num_cases = len(cases)

    return {
        "total": sum(durations),
        "avg_task": sum(task_durations) / num_cases,
        "avg_total": sum(durations) / num_cases,
        "p50": _calculate_percentile(sorted_durations, 0.50),
        "p95": _calculate_percentile(sorted_durations, 0.95),
        "p99": _calculate_percentile(sorted_durations, 0.99),
    }


def _create_metrics_table(
    accuracy: float | None,
    num_cases: int,
    latency: dict[str, float],
    tokens: dict[str, int],
) -> Table:
    """Create a Rich table with evaluation metrics.

    Args:
        accuracy: Accuracy percentage (0.0 to 1.0) or None
        num_cases: Number of evaluation cases
        latency: Dict with latency metrics
        tokens: Dict with token metrics

    Returns:
        Formatted Rich Table
    """
    table = Table(title="Evaluation Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")

    if accuracy is not None:
        table.add_row("Accuracy", f"{accuracy:.2%}")
    table.add_row("Total Cases", f"{num_cases:,}")
    table.add_row("", "")

    table.add_row("[bold yellow]Latency Metrics[/bold yellow]", "")
    table.add_row("Total Duration", f"{latency['total']:.2f}s")
    table.add_row("Avg Task Latency", f"{latency['avg_task']:.2f}s")
    table.add_row("Avg Total Latency", f"{latency['avg_total']:.2f}s")
    table.add_row("P50 Latency", f"{latency['p50']:.2f}s")
    table.add_row("P95 Latency", f"{latency['p95']:.2f}s")
    table.add_row("P99 Latency", f"{latency['p99']:.2f}s")
    table.add_row("", "")

    table.add_row("[bold yellow]Token Metrics[/bold yellow]", "")
    table.add_row("Total Input Tokens", f"{tokens['total_input']:,}")
    table.add_row("Total Output Tokens", f"{tokens['total_output']:,}")
    avg_input = tokens["total_input"] / num_cases if num_cases > 0 else 0
    avg_output = tokens["total_output"] / num_cases if num_cases > 0 else 0
    table.add_row("Avg Input Tokens/Case", f"{avg_input:.0f}")
    table.add_row("Avg Output Tokens/Case", f"{avg_output:.0f}")

    return table


def _save_results(
    run_dir: Path,
    dataset_name: str,
    config_name: str,
    limit: int | None,
    concurrency: int,
    report,
) -> None:
    """Save evaluation results to disk.

    Args:
        run_dir: Directory to save results
        dataset_name: Dataset identifier
        config_name: Config identifier
        limit: Sample limit (if any)
        concurrency: Concurrency setting
        report: Evaluation report object
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "dataset": dataset_name,
        "config": config_name,
        "limit": limit,
        "concurrency": concurrency,
        "num_cases": len(report.cases),
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    with (run_dir / "results.jsonl").open("w") as f:
        for case in report.cases:
            result = {
                "case_id": case.name,
                "output": case.output,
                "expected": case.expected_output,
                "correct": all(a.value for a in case.assertions.values()),
                "task_duration_s": case.task_duration,
                "total_duration_s": case.total_duration,
                "metrics": case.metrics,
                "metadata": case.metadata,
            }
            f.write(json.dumps(result) + "\n")

    avg = report.averages()
    tokens = _compute_token_metrics(report.cases)
    latency = _compute_latency_metrics(report.cases)

    metrics_data = {
        "accuracy": avg.assertions if avg else 0.0,
        "total_cases": len(report.cases),
        "correct": sum(1 for c in report.cases if all(a.value for a in c.assertions.values())),
        "total_input_tokens": tokens["total_input"],
        "total_output_tokens": tokens["total_output"],
        "avg_input_tokens": tokens["total_input"] / len(report.cases) if report.cases else 0,
        "avg_output_tokens": tokens["total_output"] / len(report.cases) if report.cases else 0,
        "avg_task_duration_s": avg.task_duration if avg else 0.0,
        "avg_total_duration_s": avg.total_duration if avg else 0.0,
        "p50_latency_s": latency["p50"],
        "p95_latency_s": latency["p95"],
        "p99_latency_s": latency["p99"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))


@app.command()
def main(
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config name"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit samples"),
    concurrency: int = typer.Option(
        _DEFAULT_CONCURRENCY, "--concurrency", "-j", help="Max concurrent evals"
    ),
    list_datasets: bool = typer.Option(False, "--list-datasets"),
    list_configs: bool = typer.Option(False, "--list-configs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed per-case report"),
) -> None:
    """Run memory QA evaluation."""
    if list_datasets:
        _print_available_options("Available datasets", DATASETS)
        return

    if list_configs:
        _print_available_options("Available configs", ANSWERERS)
        return

    dataset = _validate_choice(dataset, DATASETS, "dataset")
    config = _validate_choice(config, ANSWERERS, "config")

    asyncio.run(run_eval(dataset, config, limit, concurrency, verbose))


async def run_eval(
    dataset_name: str,
    config_name: str,
    limit: int | None,
    concurrency: int,
    verbose: bool,
) -> None:
    """Run evaluation using pydantic-evals."""
    print(f"\n{_SEPARATOR}")
    print("memharness evaluation")
    print(_SEPARATOR)
    print(f"Dataset:     {dataset_name}")
    print(f"Config:      {config_name}")
    if limit:
        print(f"Limit:       {limit}")
    print(f"Concurrency: {concurrency}")
    print(f"{_SEPARATOR}\n")

    try:
        dataset = DATASETS[dataset_name](limit=limit)
        print(f"Loaded {len(dataset.cases)} cases\n")

        report = await dataset.evaluate(ANSWERERS[config_name], max_concurrency=concurrency)

        timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        run_dir = Path("runs") / f"{dataset_name}-{config_name}-{timestamp}"
        _save_results(run_dir, dataset_name, config_name, limit, concurrency, report)
        print(f"Results saved to: {run_dir}\n")

        if verbose:
            print()
            report.print(include_input=False, include_output=True, include_durations=True)

        print()
        avg = report.averages()
        tokens = _compute_token_metrics(report.cases)
        latency = _compute_latency_metrics(report.cases)

        table = _create_metrics_table(
            accuracy=avg.assertions if avg else None,
            num_cases=len(report.cases),
            latency=latency,
            tokens=tokens,
        )
        Console().print(table)
        print()

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted")
        raise typer.Exit(130) from None
    except Exception as e:
        print(f"\n\nError: {e}")
        traceback.print_exc()
        raise typer.Exit(1) from e


def cli_main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    cli_main()
