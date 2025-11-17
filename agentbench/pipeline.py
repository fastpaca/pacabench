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
from agentbench.runners.base import Runner
from agentbench.types import Case, CaseResult, EvalContext, JudgeMetrics

console = Console()


async def run_case(case: Case, ctx: EvalContext) -> CaseResult:
    """
    Run complete evaluation pipeline for a single case.

    Pipeline: Case → Runner → Dataset Evaluator → CaseResult

    Args:
        case: Test case to evaluate
        ctx: Evaluation context

    Returns:
        CaseResult with evaluation results
    """
    runner_output = await ctx.runner.run_case(case, ctx)
    evaluation, judge_metrics_dict = await ctx.dataset.eval(
        case,
        runner_output.output,
        runner_output.error,
        judge_model=ctx.judge_model,
        judge_client=ctx.judge_client,
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
        metrics=runner_output.metrics,
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
    ctx = EvalContext(
        dataset=dataset,
        runner=runner,
        results=results,
        judge_model=judge_model,
        judge_client=AsyncOpenAI(),
        model=model,
        openai_api_key=openai_api_key,
        run_id=run_id,
        proxy=proxy,
        proxy_port=proxy_port,
        embedding_model=embedding_model,
    )

    console.print("[yellow]Running evaluation...[/yellow]")

    for case in tqdm(cases, desc="Evaluating cases", unit="case"):
        case_result = await run_case(case, ctx)
        ctx.results.add_case(case_result)

        if case_result.error:
            console.print(f"[red]✗ Error: {case_result.error}[/red]")

    console.print()
    console.print("[green]✓ Evaluation complete[/green]")
    console.print()

    proxy.stop()

    return results
