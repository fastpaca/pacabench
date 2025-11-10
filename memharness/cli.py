"""CLI for memharness."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from memharness.configs import ANSWERERS, DATASETS

app = typer.Typer(
    name="memharness",
    help="Memory QA benchmark harness for evaluating memory systems.",
    add_completion=False,
)


@app.command()
def main(
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config name"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit samples"),
    concurrency: int = typer.Option(10, "--concurrency", "-j", help="Max concurrent evals"),
    list_datasets: bool = typer.Option(False, "--list-datasets"),
    list_configs: bool = typer.Option(False, "--list-configs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed per-case report"),
):
    """Run memory QA evaluation."""
    # Handle list commands
    if list_datasets:
        print("Available datasets:")
        for name in sorted(DATASETS.keys()):
            print(f"  {name}")
        return

    if list_configs:
        print("Available configs:")
        for name in sorted(ANSWERERS.keys()):
            print(f"  {name}")
        return

    # Validate required args
    if dataset is None:
        print("Error: --dataset is required\n")
        print("Available datasets:")
        for name in sorted(DATASETS.keys()):
            print(f"  {name}")
        print()
        raise typer.Exit(1)

    if config is None:
        print("Error: --config is required\n")
        print("Available configs:")
        for name in sorted(ANSWERERS.keys()):
            print(f"  {name}")
        print()
        raise typer.Exit(1)

    # Validate dataset exists
    if dataset not in DATASETS:
        print(f"Error: Unknown dataset '{dataset}'\n")
        print("Available datasets:")
        for name in sorted(DATASETS.keys()):
            print(f"  {name}")
        print()
        raise typer.Exit(1)

    # Validate config exists
    if config not in ANSWERERS:
        print(f"Error: Unknown config '{config}'\n")
        print("Available configs:")
        for name in sorted(ANSWERERS.keys()):
            print(f"  {name}")
        print()
        raise typer.Exit(1)

    # Run evaluation
    asyncio.run(run_eval(dataset, config, limit, concurrency, verbose))


async def run_eval(dataset_name: str, config_name: str, limit: int | None, concurrency: int, verbose: bool):
    """Run evaluation using pydantic-evals."""
    print(f"\n{'='*60}")
    print("memharness evaluation")
    print(f"{'='*60}")
    print(f"Dataset:     {dataset_name}")
    print(f"Config:      {config_name}")
    if limit:
        print(f"Limit:       {limit}")
    print(f"Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    try:
        # Load dataset
        dataset_loader = DATASETS[dataset_name]
        dataset = dataset_loader(limit=limit)

        print(f"Loaded {len(dataset.cases)} cases\n")

        # Get task
        task = ANSWERERS[config_name]

        # Run evaluation (pydantic-evals handles everything)
        report = await dataset.evaluate(task, max_concurrency=concurrency)

        # Save results to runs directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / f"{dataset_name}-{config_name}-{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_data = {
            "dataset": dataset_name,
            "config": config_name,
            "limit": limit,
            "concurrency": concurrency,
            "num_cases": len(report.cases),
            "timestamp": datetime.now().isoformat(),
        }
        (run_dir / "config.json").write_text(json.dumps(config_data, indent=2))

        # Save per-case results as JSONL
        with (run_dir / "results.jsonl").open("w") as f:
            for case in report.cases:
                result = {
                    "case_id": case.name,
                    "output": case.output,
                    "expected": case.expected_output,
                    "correct": all(a.value for a in case.assertions.values()),
                    "task_duration_s": getattr(case, "task_duration", 0),
                    "total_duration_s": getattr(case, "total_duration", 0),
                    "metrics": case.metrics,
                    "metadata": case.metadata,
                }
                f.write(json.dumps(result) + "\n")

        # Save aggregated metrics
        avg = report.averages()
        total_input = sum(c.metrics.get("input_tokens", 0) for c in report.cases)
        total_output = sum(c.metrics.get("output_tokens", 0) for c in report.cases)

        metrics_data = {
            "accuracy": avg.assertions if avg else 0.0,
            "total_cases": len(report.cases),
            "correct": sum(1 for c in report.cases if all(a.value for a in c.assertions.values())),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_input_tokens": total_input / len(report.cases) if report.cases else 0,
            "avg_output_tokens": total_output / len(report.cases) if report.cases else 0,
            "avg_task_duration_s": getattr(avg, "task_duration", 0) if avg else 0.0,
            "avg_total_duration_s": getattr(avg, "total_duration", 0) if avg else 0.0,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))

        print(f"Results saved to: {run_dir}\n")

        # Print report (only if verbose flag is set)
        if verbose:
            print()
            report.print(include_input=False, include_output=True, include_durations=True)

        # Print aggregated metrics (always shown)
        print()
        console = Console()

        # Calculate metrics
        avg = report.averages()
        total_input = sum(case.metrics.get("input_tokens", 0) for case in report.cases)
        total_output = sum(case.metrics.get("output_tokens", 0) for case in report.cases)
        num_cases = len(report.cases) if report.cases else 1

        # Calculate latency metrics (using total_duration which includes task + evaluators)
        durations = [getattr(case, "total_duration", 0) for case in report.cases]
        task_durations = [getattr(case, "task_duration", 0) for case in report.cases]
        total_duration = sum(durations)
        avg_duration = total_duration / num_cases if num_cases else 0
        avg_task_duration = sum(task_durations) / num_cases if num_cases else 0

        # Calculate percentiles if we have durations
        if durations:
            sorted_durations = sorted(durations)
            p50_idx = int(len(sorted_durations) * 0.50)
            p95_idx = int(len(sorted_durations) * 0.95)
            p99_idx = int(len(sorted_durations) * 0.99)
            p50 = sorted_durations[p50_idx] if p50_idx < len(sorted_durations) else 0
            p95 = sorted_durations[p95_idx] if p95_idx < len(sorted_durations) else 0
            p99 = sorted_durations[p99_idx] if p99_idx < len(sorted_durations) else 0
        else:
            p50 = p95 = p99 = 0

        # Create metrics table
        table = Table(title="Evaluation Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")

        # Add accuracy and count
        if avg:
            table.add_row("Accuracy", f"{avg.assertions:.2%}")
        table.add_row("Total Cases", f"{num_cases:,}")

        # Add section separator
        table.add_row("", "")

        # Add latency metrics (IMPORTANT!)
        table.add_row("[bold yellow]Latency Metrics[/bold yellow]", "")
        table.add_row("Total Duration", f"{total_duration:.2f}s")
        table.add_row("Avg Task Latency", f"{avg_task_duration:.2f}s")
        table.add_row("Avg Total Latency", f"{avg_duration:.2f}s")
        table.add_row("P50 Latency", f"{p50:.2f}s")
        table.add_row("P95 Latency", f"{p95:.2f}s")
        table.add_row("P99 Latency", f"{p99:.2f}s")

        # Add section separator
        table.add_row("", "")

        # Add token metrics
        table.add_row("[bold yellow]Token Metrics[/bold yellow]", "")
        table.add_row("Total Input Tokens", f"{total_input:,}")
        table.add_row("Total Output Tokens", f"{total_output:,}")
        table.add_row("Avg Input Tokens/Case", f"{total_input / num_cases:.0f}")
        table.add_row("Avg Output Tokens/Case", f"{total_output / num_cases:.0f}")

        console.print(table)
        print()

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted")
        raise typer.Exit(130)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def cli_main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    cli_main()
