"""CLI for memharness."""

from pathlib import Path

import typer

from memharness.configs import ANSWERERS, DATASETS
from memharness.eval import evaluate

app = typer.Typer(
    name="memharness",
    help="Memory QA benchmark harness for evaluating memory systems.",
    add_completion=False,
)


@app.command()
def main(
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset name (e.g., 'membench', 'locomo')",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config name (e.g., 'claude-opus-long-context', 'zep-claude')",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of samples to evaluate",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (auto-generated if not specified)",
    ),
    list_datasets: bool = typer.Option(
        False,
        "--list-datasets",
        help="List available datasets and exit",
    ),
    list_configs: bool = typer.Option(
        False,
        "--list-configs",
        help="List available configs and exit",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override model name (e.g., 'gpt-4o', 'qwen/qwen3-30b-a3b-2507')",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Override base URL for custom endpoints",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="Override API key",
    ),
    concurrency: int = typer.Option(
        10,
        "--concurrency",
        "-j",
        help="Maximum concurrent evaluations (default: 10)",
    ),
):
    """Run memory QA evaluation.

    Examples:

        # Run evaluation with default settings
        memharness --dataset membench --config claude-opus-long-context

        # Override model and base URL for local model
        memharness --dataset membench --config local-long-context --model "qwen/qwen3-30b-a3b-2507" --base-url "http://localhost:1234/v1"

        # High concurrency for local model
        memharness --dataset membench --config local-long-context --concurrency 20 --limit 100

        # Limit to 100 samples
        memharness --dataset membench --config zep-claude --limit 100

        # Custom output directory
        memharness --dataset locomo --config gpt4-long-context --output-dir ./my-results

        # List available options
        memharness --list-datasets
        memharness --list-configs
    """
    # Handle list commands
    if list_datasets:
        print("Available datasets:")
        for name in sorted(DATASETS.keys()):
            dataset_obj = DATASETS[name]
            print(f"  {name:20s} {dataset_obj.__class__.__name__}")
        return

    if list_configs:
        print("Available configs:")
        for name in sorted(ANSWERERS.keys()):
            print(f"  {name}")
        return

    # Check required args
    if dataset is None:
        print("Error: --dataset is required\n")
        print("Available datasets:")
        for name in sorted(DATASETS.keys()):
            dataset_obj = DATASETS[name]
            print(f"  {name:20s} {dataset_obj.__class__.__name__}")
        print()
        raise typer.Exit(1)

    if config is None:
        print("Error: --config is required\n")
        print("Available configs:")
        for name in sorted(ANSWERERS.keys()):
            print(f"  {name}")
        print()
        raise typer.Exit(1)

    # Validate dataset
    if dataset not in DATASETS:
        print(f"Error: Unknown dataset '{dataset}'\n")
        print("Available datasets:")
        for name in sorted(DATASETS.keys()):
            dataset_obj = DATASETS[name]
            print(f"  {name:20s} {dataset_obj.__class__.__name__}")
        print()
        raise typer.Exit(1)

    # Validate config
    if config not in ANSWERERS:
        print(f"Error: Unknown config '{config}'\n")
        print("Available configs:")
        for name in sorted(ANSWERERS.keys()):
            print(f"  {name}")
        print()
        raise typer.Exit(1)

    # Get instances
    dataset_obj = DATASETS[dataset]
    answerer = ANSWERERS[config]

    # Apply CLI overrides to answerer
    if model is not None:
        answerer.model = model
        answerer._agent = None  # Reset lazy-loaded agent
    if base_url is not None:
        answerer.base_url = base_url
        answerer._agent = None  # Reset lazy-loaded agent
    if api_key is not None:
        answerer.api_key = api_key
        answerer._agent = None  # Reset lazy-loaded agent

    # Convert output_dir to Path if provided
    output_path = Path(output_dir) if output_dir else None

    # Run evaluation
    print(f"\n{'='*60}")
    print("memharness evaluation")
    print(f"{'='*60}")
    print(f"Dataset:  {dataset}")
    print(f"Config:   {config}")
    if model:
        print(f"Model:    {model}")
    if base_url:
        print(f"Base URL: {base_url}")
    if limit:
        print(f"Limit:    {limit}")
    print(f"Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    try:
        metrics = evaluate(
            dataset=dataset_obj,
            answerer=answerer,
            output_dir=output_path,
            limit=limit,
            concurrency=concurrency,
        )

        # Print summary
        print(f"\n{'='*60}")
        print("Evaluation Complete")
        print(f"{'='*60}")
        print(f"Accuracy:              {metrics.accuracy:.2%}")
        print(f"Correct:               {metrics.correct}/{metrics.total_samples}")
        print(f"Avg Latency:           {metrics.avg_latency_ms:.0f}ms")
        print(f"P95 Latency:           {metrics.p95_latency_ms:.0f}ms")
        print(f"Avg Input Tokens:      {metrics.avg_input_tokens:.0f}")
        print(f"Avg Output Tokens:     {metrics.avg_output_tokens:.0f}")
        print(f"Total Input Tokens:    {metrics.total_input_tokens:,}")
        print(f"Total Output Tokens:   {metrics.total_output_tokens:,}")

        if metrics.custom_metrics:
            print("\nCustom Metrics:")
            for key, value in sorted(metrics.custom_metrics.items()):
                if isinstance(value, float):
                    print(f"  {key:25s} {value:.2f}")
                else:
                    print(f"  {key:25s} {value}")

        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        raise typer.Exit(130)
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        raise typer.Exit(1)


def cli_main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    cli_main()
