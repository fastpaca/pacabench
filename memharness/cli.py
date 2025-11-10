"""CLI for memharness."""

import asyncio
import json
import re
import traceback
from datetime import datetime
from pathlib import Path

import questionary
import typer
from rich.console import Console
from rich.table import Table

from memharness.configs import ANSWERERS, DATASETS

# Constants
_DEFAULT_CONCURRENCY = 10
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


def _parse_config_list(config_str: str | None) -> list[str]:
    """Parse config string into list of config names.

    Args:
        config_str: Comma-separated config names or single config

    Returns:
        List of config names
    """
    if not config_str:
        return []

    configs = [c.strip() for c in config_str.split(",")]
    return [c for c in configs if c]


def _prompt_config_multiselect() -> list[str]:
    """Show interactive TUI to select multiple configs.

    Returns:
        List of selected config names
    """
    choices = sorted(ANSWERERS.keys())
    selected = questionary.checkbox(
        "Select configurations to battle (Space to select, Enter to confirm):",
        choices=choices,
    ).ask()

    if not selected:
        print("No configurations selected")
        raise typer.Exit(1)

    return selected


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


def _compute_f1_metrics(cases: list) -> dict[str, float]:
    """Compute F1 score metrics across all cases.

    Args:
        cases: List of report cases

    Returns:
        Dict with F1 statistics
    """
    f1_scores = []

    for case in cases:
        response = case.output.strip().lower() if case.output else ""
        expected = case.expected_output.strip().lower() if case.expected_output else ""

        if not expected or not response:
            f1_scores.append(0.0)
            continue

        response_tokens = set(_tokenize_text(response))
        expected_tokens = set(_tokenize_text(expected))

        if not expected_tokens:
            f1_scores.append(0.0)
            continue

        overlap = response_tokens & expected_tokens
        if not overlap:
            f1_scores.append(0.0)
            continue

        precision = len(overlap) / len(response_tokens) if response_tokens else 0.0
        recall = len(overlap) / len(expected_tokens) if expected_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    if not f1_scores:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}

    return {
        "avg": sum(f1_scores) / len(f1_scores),
        "min": min(f1_scores),
        "max": max(f1_scores),
    }


def _tokenize_text(text: str) -> list[str]:
    """Simple whitespace tokenization with punctuation handling."""
    return [token for token in re.findall(r"\b\w+\b", text.lower()) if token]


def _create_metrics_table(
    precision: float | None,
    num_cases: int,
    latency: dict[str, float],
    tokens: dict[str, int],
    f1: dict[str, float] | None = None,
) -> Table:
    """Create a Rich table with evaluation metrics.

    Args:
        precision: Precision percentage (0.0 to 1.0) or None
        num_cases: Number of evaluation cases
        latency: Dict with latency metrics
        tokens: Dict with token metrics
        f1: Dict with F1 score metrics (optional)

    Returns:
        Formatted Rich Table
    """
    table = Table(title="Evaluation Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")

    if precision is not None:
        table.add_row("Precision", f"{precision:.2%}")
    table.add_row("Total Cases", f"{num_cases:,}")
    table.add_row("", "")

    if f1:
        table.add_row("[bold yellow]F1 Score Metrics[/bold yellow]", "")
        table.add_row("Avg F1 Score", f"{f1['avg']:.4f}")
        table.add_row("Min F1 Score", f"{f1['min']:.4f}")
        table.add_row("Max F1 Score", f"{f1['max']:.4f}")
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
    f1 = _compute_f1_metrics(report.cases)

    metrics_data = {
        "precision": avg.assertions if avg else 0.0,
        "total_cases": len(report.cases),
        "correct": sum(1 for c in report.cases if all(a.value for a in c.assertions.values())),
        "f1_avg": f1["avg"],
        "f1_min": f1["min"],
        "f1_max": f1["max"],
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


def _create_leaderboard_table(results: list[dict]) -> Table:
    """Create leaderboard-style comparison table.

    Args:
        results: List of dicts with config_name and metrics

    Returns:
        Formatted Rich Table
    """
    table = Table(title="Battle Results", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", justify="center", no_wrap=True)
    table.add_column("Config", style="white")
    table.add_column("Precision", style="green", justify="right")
    table.add_column("Avg F1", style="yellow", justify="right")
    table.add_column("P50", style="blue", justify="right")
    table.add_column("P95", style="blue", justify="right")
    table.add_column("Avg Input", style="magenta", justify="right")
    table.add_column("Avg Output", style="magenta", justify="right")

    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            result["config_name"],
            f"{result['precision']:.2%}",
            f"{result['f1_avg']:.4f}",
            f"{result['p50_latency']:.2f}s",
            f"{result['p95_latency']:.2f}s",
            f"{result['avg_input_tokens']:.0f}",
            f"{result['avg_output_tokens']:.0f}",
        )

    return table


async def _validate_config(config_name: str, test_case: dict) -> tuple[bool, str | None]:
    """Validate that a config can run successfully.

    Args:
        config_name: Name of config to validate
        test_case: Sample test case to run

    Returns:
        Tuple of (success, error_message)
    """
    try:
        task = ANSWERERS[config_name]
        await task(test_case)
        return True, None
    except Exception as e:
        error_msg = str(e)
        if "api" in error_msg.lower() and "key" in error_msg.lower():
            return False, f"API key missing or invalid: {error_msg}"
        return False, f"Configuration error: {error_msg}"


async def run_battle(
    dataset_name: str,
    config_names: list[str],
    limit: int | None,
    concurrency: int,
    verbose: bool,
) -> None:
    """Run battle mode evaluation across multiple configs.

    Args:
        dataset_name: Dataset to evaluate
        config_names: List of config names to battle
        limit: Optional limit on number of samples
        concurrency: Max concurrent evaluations
        verbose: Whether to show detailed per-case reports
    """
    console = Console()
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="dim", justify="right")
    info_table.add_column(style="white")

    info_table.add_row("Dataset:", dataset_name)
    info_table.add_row("Configs:", ", ".join(config_names))
    if limit:
        info_table.add_row("Limit:", str(limit))
    info_table.add_row("Concurrency:", str(concurrency))

    console.print()
    console.print("[bold magenta]Battle Mode[/bold magenta]")
    console.print(info_table)
    console.print()

    try:
        dataset = DATASETS[dataset_name](limit=limit)
        print(f"Loaded {len(dataset.cases)} cases\n")

        if dataset.cases:
            print("Validating configurations...")
            test_case = dataset.cases[0].inputs
            valid_configs = []
            failed_configs = []

            for config_name in config_names:
                success, error = await _validate_config(config_name, test_case)
                if success:
                    valid_configs.append(config_name)
                    print(f"  [OK] {config_name}")
                else:
                    failed_configs.append((config_name, error))
                    print(f"  [FAIL] {config_name}: {error}")

            if not valid_configs:
                print("\nError: All configurations failed validation")
                raise typer.Exit(1)

            if failed_configs:
                print(
                    f"\nWarning: Skipping {len(failed_configs)} failed config(s), continuing with {len(valid_configs)}\n"
                )

            config_names = valid_configs

        results = []
        timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        battle_dir = Path("runs") / f"battle-{dataset_name}-{timestamp}"
        battle_dir.mkdir(parents=True, exist_ok=True)

        for config_name in config_names:
            print(f"Evaluating {config_name}...")
            try:
                report = await dataset.evaluate(ANSWERERS[config_name], max_concurrency=concurrency)
            except Exception as e:
                print(f"  [FAIL] Evaluation failed: {e}\n")
                continue

            run_dir = battle_dir / config_name
            _save_results(run_dir, dataset_name, config_name, limit, concurrency, report)

            avg = report.averages()
            tokens = _compute_token_metrics(report.cases)
            latency = _compute_latency_metrics(report.cases)
            f1 = _compute_f1_metrics(report.cases)

            results.append(
                {
                    "config_name": config_name,
                    "precision": avg.assertions if avg else 0.0,
                    "f1_avg": f1["avg"],
                    "p50_latency": latency["p50"],
                    "p95_latency": latency["p95"],
                    "avg_input_tokens": tokens["total_input"] / len(report.cases)
                    if report.cases
                    else 0,
                    "avg_output_tokens": tokens["total_output"] / len(report.cases)
                    if report.cases
                    else 0,
                    "report": report,
                }
            )

        if not results:
            print("\nError: No configurations completed evaluation successfully")
            raise typer.Exit(1)

        results.sort(key=lambda x: (-x["precision"], -x["f1_avg"], x["p50_latency"]))

        print(f"\n\nBattle results saved to: {battle_dir}\n")

        leaderboard = _create_leaderboard_table(results)
        Console().print(leaderboard)
        print()

        battle_summary = {
            "dataset": dataset_name,
            "timestamp": timestamp,
            "num_cases": len(dataset.cases),
            "configs": [
                {
                    "rank": i + 1,
                    "config_name": r["config_name"],
                    "precision": r["precision"],
                    "f1_avg": r["f1_avg"],
                    "p50_latency": r["p50_latency"],
                    "p95_latency": r["p95_latency"],
                    "avg_input_tokens": r["avg_input_tokens"],
                    "avg_output_tokens": r["avg_output_tokens"],
                }
                for i, r in enumerate(results)
            ],
        }
        (battle_dir / "battle_summary.json").write_text(json.dumps(battle_summary, indent=2))

        if verbose:
            print("\nDetailed results per config:\n")
            for result in results:
                print(f"\n{result['config_name']}:")
                result["report"].print(
                    include_input=False, include_output=True, include_durations=True
                )

    except KeyboardInterrupt:
        print("\n\nBattle interrupted")
        raise typer.Exit(130) from None
    except Exception as e:
        print(f"\n\nError: {e}")
        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def main(
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    config: str | None = typer.Option(
        None, "--config", "-c", help="Config name(s) (comma-separated or multiple -c flags)"
    ),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit samples"),
    concurrency: int = typer.Option(
        _DEFAULT_CONCURRENCY, "--concurrency", "-j", help="Max concurrent evals"
    ),
    list_datasets: bool = typer.Option(False, "--list-datasets"),
    list_configs: bool = typer.Option(False, "--list-configs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed per-case report"),
) -> None:
    """Run memory QA evaluation (single or battle mode)."""
    if list_datasets:
        _print_available_options("Available datasets", DATASETS)
        return

    if list_configs:
        _print_available_options("Available configs", ANSWERERS)
        return

    dataset = _validate_choice(dataset, DATASETS, "dataset")

    config_list = _parse_config_list(config)

    if not config_list:
        config_list = _prompt_config_multiselect()

    for cfg in config_list:
        if cfg not in ANSWERERS:
            print(f"Error: Unknown config '{cfg}'\n")
            _print_available_options("Available configs", ANSWERERS)
            raise typer.Exit(1)

    if len(config_list) == 1:
        asyncio.run(run_eval(dataset, config_list[0], limit, concurrency, verbose))
    else:
        asyncio.run(run_battle(dataset, config_list, limit, concurrency, verbose))


async def run_eval(
    dataset_name: str,
    config_name: str,
    limit: int | None,
    concurrency: int,
    verbose: bool,
) -> None:
    """Run evaluation using pydantic-evals."""
    console = Console()
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="dim", justify="right")
    info_table.add_column(style="white")

    info_table.add_row("Dataset:", dataset_name)
    info_table.add_row("Config:", config_name)
    if limit:
        info_table.add_row("Limit:", str(limit))
    info_table.add_row("Concurrency:", str(concurrency))

    console.print()
    console.print("[bold cyan]Evaluation[/bold cyan]")
    console.print(info_table)
    console.print()

    try:
        dataset = DATASETS[dataset_name](limit=limit)
        print(f"Loaded {len(dataset.cases)} cases\n")

        if dataset.cases:
            print("Validating configuration...")
            test_case = dataset.cases[0].inputs
            success, error = await _validate_config(config_name, test_case)
            if not success:
                print(f"[FAIL] Configuration validation failed: {error}\n")
                raise typer.Exit(1)
            print("[OK] Configuration validated\n")

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
        f1 = _compute_f1_metrics(report.cases)

        table = _create_metrics_table(
            precision=avg.assertions if avg else None,
            num_cases=len(report.cases),
            latency=latency,
            tokens=tokens,
            f1=f1,
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
