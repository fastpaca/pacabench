"""AgentBench CLI - Process-based benchmark harness."""

import os
from datetime import datetime
from pathlib import Path

import typer
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from agentbench import datasets, evaluators
from agentbench.metrics import CaseResult, aggregate_results, save_results
from agentbench.proxy import ProxyServer
from agentbench.runner import spawn_runner

app = typer.Typer()
console = Console()


@app.command()
def main(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset name (membench, longmemeval, gaia)"
    ),
    runner: str = typer.Option(
        ..., "--runner", "-r", help="Runner path (e.g., qa/long_context, agentic/mem0)"
    ),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model name"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of cases"),
    split: str | None = typer.Option(None, "--split", "-s", help="Dataset split"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Run AgentBench evaluation."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)

    console.print("[bold]AgentBench[/bold]")
    console.print(f"Dataset: {dataset}")
    console.print(f"Runner: {runner}")
    console.print(f"Model: {model}")
    console.print()

    console.print("[yellow]Starting LLM proxy server...[/yellow]")
    proxy = ProxyServer(port=8000, openai_api_key=openai_api_key)
    proxy.start()
    console.print("[green]✓ Proxy server started on http://localhost:8000[/green]")
    console.print()

    console.print(f"[yellow]Loading {dataset} dataset...[/yellow]")
    cases = _load_dataset(dataset, split, limit)
    console.print(f"[green]✓ Loaded {len(cases)} cases[/green]")
    console.print()

    judge_client = OpenAI()

    console.print("[yellow]Running evaluation...[/yellow]")
    results: list[CaseResult] = []

    for idx, case in enumerate(cases, 1):
        console.print(f"[dim]Case {idx}/{len(cases)}: {case.id}[/dim]")

        env = {
            "MODEL": model,
            "OPENAI_API_KEY": openai_api_key,
            "OPENAI_BASE_URL": "http://localhost:8000/v1",
            "PATH": os.environ.get("PATH", ""),
        }

        runner_result = spawn_runner(
            runner_script=f"runners/{runner}.py",
            case_id=case.id,
            task_type=case.task_type,
            inputs=case.inputs,
            env=env,
        )

        llm_metrics = proxy.metrics.get_metrics("_current")

        if runner_result.error:
            console.print(f"[red]✗ Error: {runner_result.error}[/red]")
            results.append(
                CaseResult(
                    case_id=case.id,
                    passed=False,
                    output=runner_result.result,
                    error=runner_result.error,
                    runner_duration_ms=runner_result.duration_ms,
                    llm_metrics=llm_metrics,
                )
            )
            continue

        passed, f1_score, f1_passed, judge_passed, judge_metrics = _evaluate_case(
            case=case,
            output=runner_result.result,
            judge_client=judge_client,
        )

        results.append(
            CaseResult(
                case_id=case.id,
                passed=passed,
                output=runner_result.result,
                error=runner_result.error,
                runner_duration_ms=runner_result.duration_ms,
                llm_metrics=llm_metrics,
                f1_score=f1_score,
                f1_passed=f1_passed,
                judge_passed=judge_passed,
                judge_metrics=judge_metrics,
            )
        )

        proxy.metrics.clear_metrics("_current")

    console.print()
    console.print("[green]✓ Evaluation complete[/green]")
    console.print()

    metrics = aggregate_results(results)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"{dataset}-{runner.replace('/', '-')}-{timestamp}"

    config = {
        "dataset": dataset,
        "runner": runner,
        "model": model,
        "limit": limit,
        "split": split,
        "timestamp": datetime.now().isoformat(),
    }

    save_results(output_dir, results, metrics, config)
    console.print(f"[green]✓ Results saved to {output_dir}[/green]")
    console.print()

    _print_metrics_table(metrics)

    proxy.stop()


def _load_dataset(name: str, split: str | None, limit: int | None) -> list:
    """Load dataset by name."""
    if name == "membench":
        return datasets.load_membench(
            split=split or "eval",
            limit=limit,
        )
    elif name == "longmemeval":
        return datasets.load_longmemeval(
            split=split or "s_cleaned",
            limit=limit,
        )
    elif name == "gaia":
        return datasets.load_gaia(
            level=split or "all",
            split="validation",
            limit=limit,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _evaluate_case(
    case, output: str | None, judge_client: OpenAI
) -> tuple[bool, float | None, bool | None, bool | None, dict | None]:
    """Evaluate a single case."""
    if not output:
        return False, None, None, None, None

    if case.task_type == "qa":
        if "choices" in case.inputs:
            passed = evaluators.evaluate_multiple_choice(output, case.expected_output, case.inputs)
            return passed, None, passed, None, None
        else:
            passed_f1, f1_score = evaluators.evaluate_f1_score(
                output, case.expected_output, case.inputs
            )
            passed_judge, judge_metrics = evaluators.evaluate_llm_judge(
                output,
                case.expected_output,
                case.inputs,
                openai_client=judge_client,
            )
            return passed_f1 and passed_judge, f1_score, passed_f1, passed_judge, judge_metrics

    elif case.task_type == "agentic":
        passed, judge_metrics = evaluators.evaluate_gaia(
            output,
            case.expected_output,
            case.inputs,
            openai_client=judge_client,
        )
        return passed, None, None, passed, judge_metrics

    return False, None, None, None, None


def _print_metrics_table(metrics) -> None:
    """Print metrics in a rich table."""
    table = Table(title="Evaluation Metrics")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Precision", f"{metrics.precision:.2%}")
    table.add_row("Accuracy (All evaluators)", f"{metrics.accuracy:.2%}")
    table.add_row("Passed Cases", f"{metrics.passed_cases}/{metrics.total_cases}")
    table.add_row("", "")
    table.add_row("Avg Duration", f"{metrics.avg_duration_s:.2f}s")
    table.add_row("P50 Duration", f"{metrics.p50_duration_s:.2f}s")
    table.add_row("P95 Duration", f"{metrics.p95_duration_s:.2f}s")
    table.add_row("", "")
    table.add_row("Total LLM Calls", str(metrics.total_llm_calls))
    table.add_row("Total Input Tokens", f"{metrics.total_input_tokens:,}")
    table.add_row("Total Output Tokens", f"{metrics.total_output_tokens:,}")
    table.add_row("Avg Input Tokens", f"{metrics.avg_input_tokens:.0f}")
    table.add_row("Avg Output Tokens", f"{metrics.avg_output_tokens:.0f}")
    table.add_row("", "")
    table.add_row("Avg LLM Latency", f"{metrics.avg_llm_latency_ms:.0f}ms")
    table.add_row("P50 LLM Latency", f"{metrics.p50_llm_latency_ms:.0f}ms")
    table.add_row("P95 LLM Latency", f"{metrics.p95_llm_latency_ms:.0f}ms")
    table.add_row("", "")
    table.add_row("Total Cost", f"${metrics.total_cost_usd:.4f}")
    table.add_row("Avg Cost/Case", f"${metrics.avg_cost_per_case_usd:.4f}")

    if metrics.avg_f1_score is not None:
        table.add_row("", "")
        table.add_row("Avg F1 Score", f"{metrics.avg_f1_score:.2f}")

    if metrics.total_judge_input_tokens is not None:
        table.add_row("", "")
        table.add_row("Judge Input Tokens", f"{metrics.total_judge_input_tokens:,}")
        table.add_row("Judge Output Tokens", f"{metrics.total_judge_output_tokens:,}")

    console.print(table)


def cli_main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    app()
