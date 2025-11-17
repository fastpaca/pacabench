"""Evaluation pipeline orchestration - Case → Runner → Evaluator → Result."""

import os
from dataclasses import dataclass
from typing import Any

import portpicker
from openai import AsyncOpenAI
from rich.console import Console
from tqdm import tqdm

from agentbench.proxy import ProxyServer
from agentbench.stages import (
    Case,
    CaseResult,
    EvaluationOutput,
    RunnerOutput,
    evaluate_f1_score,
    evaluate_gaia,
    evaluate_llm_judge,
    evaluate_multiple_choice,
    spawn_runner,
)

console = Console()


@dataclass
class EvalContext:
    """Evaluation context carrying shared state across pipeline stages."""

    runner_path: str
    model: str
    openai_api_key: str
    run_id: str
    dataset: str
    proxy: ProxyServer
    proxy_port: int
    judge_client: AsyncOpenAI
    judge_model: str = "gpt-4o-mini"
    embedding_model: str | None = None


def build_env(case: Case, ctx: EvalContext) -> dict[str, str]:
    """Stage 1: Build runner environment."""
    env = {
        "MODEL": ctx.model,
        "OPENAI_API_KEY": ctx.openai_api_key,
        "OPENAI_BASE_URL": f"http://localhost:{ctx.proxy_port}/v1",
        "PATH": os.environ.get("PATH", ""),
        "AGENTBENCH_RUN_ID": ctx.run_id,
        "AGENTBENCH_DATASET": ctx.dataset,
    }
    if ctx.embedding_model:
        env["EMBEDDING_MODEL"] = ctx.embedding_model
    return env


async def run_runner(case: Case, env: dict[str, str], ctx: EvalContext) -> RunnerOutput:
    """Stage 2: Execute runner."""
    return await spawn_runner(
        case=case,
        runner_script=f"runners/{ctx.runner_path}.py",
        env=env,
    )


def collect_metrics(ctx: EvalContext) -> dict[str, Any]:
    """Stage 3: Collect LLM metrics from proxy."""
    metrics = ctx.proxy.metrics.get_metrics("_current")
    ctx.proxy.metrics.clear_metrics("_current")
    return metrics


async def evaluate(case: Case, runner_output: RunnerOutput, ctx: EvalContext) -> EvaluationOutput:
    """Stage 4: Evaluate runner output."""
    if runner_output.error:
        return EvaluationOutput(passed=False)

    if not runner_output.result:
        return EvaluationOutput(passed=False)

    if case.task_type == "qa":
        if "choices" in case.inputs:
            return evaluate_multiple_choice(case, runner_output)
        else:
            f1_output = evaluate_f1_score(case, runner_output)
            judge_output = await evaluate_llm_judge(
                case,
                runner_output,
                model=ctx.judge_model,
                openai_client=ctx.judge_client,
            )
            return EvaluationOutput(
                passed=f1_output.f1_passed and judge_output.judge_passed,
                f1_score=f1_output.f1_score,
                f1_passed=f1_output.f1_passed,
                judge_passed=judge_output.judge_passed,
                judge_metrics=judge_output.judge_metrics,
            )

    elif case.task_type == "agentic":
        return await evaluate_gaia(
            case,
            runner_output,
            model=ctx.judge_model,
            openai_client=ctx.judge_client,
        )

    return EvaluationOutput(passed=False)


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
    env = build_env(case, ctx)
    runner_output = await run_runner(case, env, ctx)
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


async def run_evaluation(
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
