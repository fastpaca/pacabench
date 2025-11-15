"""Evaluation pipeline orchestration - Case → Runner → Evaluator → Result."""

import os

from openai import OpenAI
from rich.console import Console

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


def evaluate_case(
    case: Case,
    runner_path: str,
    model: str,
    openai_api_key: str,
    run_id: str,
    dataset: str,
    proxy: ProxyServer,
    judge_client: OpenAI,
    judge_model: str,
    embedding_model: str | None = None,
) -> CaseResult:
    """
    Run complete evaluation pipeline for a single case.

    Pipeline: Case → Runner → Evaluator → Result

    Args:
        case: Test case to evaluate
        runner_path: Path to runner script
        model: Model name
        openai_api_key: OpenAI API key
        run_id: Run identifier
        dataset: Dataset name
        proxy: Proxy server for metrics collection
        judge_client: OpenAI client for judge
        judge_model: Judge model name
        embedding_model: Embedding model name (optional)

    Returns:
        CaseResult with evaluation results
    """
    env = {
        "MODEL": model,
        "OPENAI_API_KEY": openai_api_key,
        "OPENAI_BASE_URL": "http://localhost:8000/v1",
        "PATH": os.environ.get("PATH", ""),
        "AGENTBENCH_RUN_ID": run_id,
        "AGENTBENCH_DATASET": dataset,
    }
    if embedding_model:
        env["EMBEDDING_MODEL"] = embedding_model

    runner_output = spawn_runner(
        case=case,
        runner_script=f"runners/{runner_path}.py",
        env=env,
    )

    llm_metrics = proxy.metrics.get_metrics("_current")
    proxy.metrics.clear_metrics("_current")

    if runner_output.error:
        eval_output = EvaluationOutput(passed=False)
    else:
        eval_output = _evaluate(
            case=case,
            runner_output=runner_output,
            judge_client=judge_client,
            judge_model=judge_model,
        )

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


def run_evaluation(
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
    console.print("[yellow]Starting LLM proxy server...[/yellow]")
    proxy = ProxyServer(
        port=8000,
        openai_api_key=openai_api_key,
        upstream_base_url=upstream_base_url,
    )
    proxy.start()
    console.print("[green]✓ Proxy server started on http://localhost:8000[/green]")
    console.print()

    judge_client = OpenAI()

    console.print("[yellow]Running evaluation...[/yellow]")
    results: list[CaseResult] = []

    for idx, case in enumerate(cases, 1):
        console.print(f"[dim]Case {idx}/{len(cases)}: {case.id}[/dim]")

        result = evaluate_case(
            case=case,
            runner_path=runner_path,
            model=model,
            openai_api_key=openai_api_key,
            run_id=run_id,
            dataset=dataset,
            proxy=proxy,
            judge_client=judge_client,
            judge_model=judge_model,
            embedding_model=embedding_model,
        )
        results.append(result)

        if result.error:
            console.print(f"[red]✗ Error: {result.error}[/red]")

    console.print()
    console.print("[green]✓ Evaluation complete[/green]")
    console.print()

    proxy.stop()

    return results


def _evaluate(
    case: Case,
    runner_output: RunnerOutput,
    judge_client: OpenAI,
    judge_model: str,
) -> EvaluationOutput:
    """
    Dispatch to appropriate evaluator based on task type.

    Args:
        case: Test case
        runner_output: Runner output
        judge_client: OpenAI client for judge
        judge_model: Judge model name

    Returns:
        EvaluationOutput with evaluation results
    """
    if not runner_output.result:
        return EvaluationOutput(passed=False)

    if case.task_type == "qa":
        if "choices" in case.inputs:
            return evaluate_multiple_choice(case, runner_output)
        else:
            f1_output = evaluate_f1_score(case, runner_output)
            judge_output = evaluate_llm_judge(
                case,
                runner_output,
                model=judge_model,
                openai_client=judge_client,
            )
            return EvaluationOutput(
                passed=f1_output.f1_passed and judge_output.judge_passed,
                f1_score=f1_output.f1_score,
                f1_passed=f1_output.f1_passed,
                judge_passed=judge_output.judge_passed,
                judge_metrics=judge_output.judge_metrics,
            )

    elif case.task_type == "agentic":
        return evaluate_gaia(
            case,
            runner_output,
            model=judge_model,
            openai_client=judge_client,
        )

    return EvaluationOutput(passed=False)
