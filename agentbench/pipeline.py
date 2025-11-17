"""Evaluation pipeline orchestration - Case → Runner → Evaluator → Result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import portpicker
from openai import AsyncOpenAI
from rich.console import Console
from tqdm import tqdm

from agentbench.datasets.base import Dataset
from agentbench.proxy import ProxyServer
from agentbench.results import Results
from agentbench.types import Case, CaseResult, JudgeMetrics, Runner, RunnerContext, RunnerMetrics

console = Console()


async def run_case(
    case: Case,
    runner: Runner,
    runner_ctx: RunnerContext,
    dataset: Dataset,
    proxy: ProxyServer,
    judge_model: str,
    judge_client: AsyncOpenAI,
) -> CaseResult:
    """
    Run complete evaluation pipeline for a single case.

    Pipeline: Case → Runner → Dataset Evaluator → CaseResult

    Args:
        case: Test case to evaluate
        runner: Runner instance
        runner_ctx: Runner execution context
        dataset: Dataset instance
        proxy: Proxy server instance
        judge_model: Judge model name
        judge_client: OpenAI client for judge

    Returns:
        CaseResult with evaluation results
    """
    runner_output = await runner.run_case(case, runner_ctx)

    llm_metrics = proxy.metrics.get_metrics("_current")
    metrics = RunnerMetrics(
        model_duration_ms=runner_output.duration_ms,
        llm_metrics=llm_metrics,
    )

    evaluation, judge_metrics_dict = await dataset.eval(
        case,
        runner_output.output,
        runner_output.error,
        judge_model=judge_model,
        judge_client=judge_client,
    )
    judge_metrics = None
    if judge_metrics_dict:
        judge_metrics = JudgeMetrics(
            input_tokens=judge_metrics_dict["input_tokens"],
            output_tokens=judge_metrics_dict["output_tokens"],
        )
    return CaseResult(
        case_id=case.id,
        output=runner_output.output,
        error=runner_output.error,
        metrics=metrics,
        evaluation=evaluation,
        judge_metrics=judge_metrics,
    )


async def run(
    dataset: Dataset,
    runner: Runner,
    model: str,
    openai_api_key: str,
    run_id: str,
    output_dir: Path,
    config: dict[str, Any],
    embedding_model: str | None = None,
    judge_model: str = "gpt-4o-mini",
    upstream_base_url: str | None = None,
    limit: int | None = None,
) -> Results:
    """
    Run evaluation pipeline on a dataset using the given runner.

    This is the main library entry point:
    Dataset → load cases → Runner → Dataset.eval → Results container.

    Returns:
        Results object containing all case results and metrics for this run.
    """
    console.print("[yellow]Loading dataset cases...[/yellow]")
    cases_iter = await dataset.load(limit=limit)
    cases = list(cases_iter)
    console.print(f"[green]✓ Loaded {len(cases)} cases[/green]")
    console.print()

    proxy_port = portpicker.pick_unused_port()
    console.print(f"[yellow]Starting LLM proxy server on port {proxy_port}...[/yellow]")
    proxy = ProxyServer(
        port=proxy_port,
        openai_api_key=openai_api_key,
        upstream_base_url=upstream_base_url,
    )
    proxy.start()
    console.print(f"[green]✓ Proxy server started on http://localhost:{proxy_port}[/green]")
    console.print()

    results = Results(output_dir=output_dir, config=config, run_id=run_id)
    runner_ctx = RunnerContext(
        model=model,
        proxy_port=proxy_port,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
    )
    judge_client = AsyncOpenAI()

    console.print("[yellow]Running evaluation...[/yellow]")

    for case in tqdm(cases, desc="Evaluating cases", unit="case"):
        case_result = await run_case(
            case=case,
            runner=runner,
            runner_ctx=runner_ctx,
            dataset=dataset,
            proxy=proxy,
            judge_model=judge_model,
            judge_client=judge_client,
        )
        results.add_case(case_result)

        if case_result.error:
            console.print(f"[red]✗ Error: {case_result.error}[/red]")

    console.print()
    console.print("[green]✓ Evaluation complete[/green]")
    console.print()

    proxy.stop()
    results.finalize()

    return results
