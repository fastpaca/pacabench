"""Evaluation pipeline orchestration - Case → Runner → Evaluator → Result."""

import portpicker
from openai import AsyncOpenAI
from rich.console import Console
from tqdm import tqdm

from agentbench.context import EvalContext
from agentbench.proxy import ProxyServer
from agentbench.stages import Case, CaseResult
from agentbench.stages.evaluator import run as evaluate
from agentbench.stages.result import collect_metrics
from agentbench.stages.runner import run as run_runner

console = Console()


async def evaluate_case(case: Case, ctx: EvalContext) -> CaseResult:
    """
    Run complete evaluation pipeline for a single case.

    Pipeline: Case → Env → Runner → Metrics → Evaluator → Result

    Args:
        case: Test case to evaluate
        ctx: Evaluation context

    Returns:
        CaseResult with evaluation results
    """
    runner_output = await run_runner(case, ctx)
    llm_metrics = collect_metrics(ctx)
    eval_output = await evaluate(case, runner_output, ctx)
    return CaseResult(
        case_id=case.id,
        passed=eval_output.passed,
        output=runner_output.result,
        error=runner_output.error,
        runner_duration_ms=runner_output.duration_ms,
        llm_metrics=llm_metrics,
        f1_score=eval_output.f1_score,
        f1_passed=eval_output.f1_passed,
        judge_passed=eval_output.judge_passed,
        judge_metrics=eval_output.judge_metrics,
    )


async def run(
    cases: list[Case],
    runner_path: str,
    model: str,
    openai_api_key: str,
    run_id: str,
    dataset: str,
    embedding_model: str | None = None,
    judge_model: str = "gpt-4o-mini",
    upstream_base_url: str | None = None,
) -> list[CaseResult]:
    """
    Run evaluation pipeline on cases.

    Args:
        cases: Test cases to evaluate
        runner_path: Path to runner script
        model: Model name
        openai_api_key: OpenAI API key
        run_id: Run identifier
        dataset: Dataset name
        embedding_model: Embedding model name (optional)
        judge_model: Judge model name
        upstream_base_url: Upstream base URL for proxy

    Returns:
        List of case results
    """
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

    ctx = EvalContext(
        runner_path=runner_path,
        model=model,
        openai_api_key=openai_api_key,
        run_id=run_id,
        dataset=dataset,
        proxy=proxy,
        proxy_port=proxy_port,
        judge_client=AsyncOpenAI(),
        judge_model=judge_model,
        embedding_model=embedding_model,
    )

    console.print("[yellow]Running evaluation...[/yellow]")
    results: list[CaseResult] = []

    for case in tqdm(cases, desc="Evaluating cases", unit="case"):
        result = await evaluate_case(case, ctx)
        results.append(result)

        if result.error:
            console.print(f"[red]✗ Error: {result.error}[/red]")

    console.print()
    console.print("[green]✓ Evaluation complete[/green]")
    console.print()

    proxy.stop()

    return results
