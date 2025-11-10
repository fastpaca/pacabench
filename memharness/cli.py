"""CLI for memharness."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import typer

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
    asyncio.run(run_eval(dataset, config, limit, concurrency))


async def run_eval(dataset_name: str, config_name: str, limit: int | None, concurrency: int):
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
                    "duration_s": getattr(case, "duration_s", getattr(case, "duration", 0)),
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
            "avg_duration_s": getattr(avg, "duration_s", getattr(avg, "duration", 0)) if avg else 0.0,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))

        print(f"Results saved to: {run_dir}\n")

        # Print report
        print()
        report.print(include_input=False, include_output=True, include_durations=True)

        # Print aggregated metrics
        print(f"\n{'='*60}")
        print("Metrics")
        print(f"{'='*60}")

        avg = report.averages()
        if avg:
            print(f"Accuracy:         {avg.assertions:.2%}")

        # Aggregate token metrics from cases
        total_input = sum(case.metrics.get("input_tokens", 0) for case in report.cases)
        total_output = sum(case.metrics.get("output_tokens", 0) for case in report.cases)
        num_cases = len(report.cases) if report.cases else 1

        print(f"Total Input:      {total_input:,} tokens")
        print(f"Total Output:     {total_output:,} tokens")
        print(f"Avg Input:        {total_input / num_cases:.0f} tokens/case")
        print(f"Avg Output:       {total_output / num_cases:.0f} tokens/case")

        print(f"{'='*60}\n")

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
