"""Evaluation pipeline orchestration - Case → Runner → Evaluator → Result."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import portpicker
from openai import AsyncOpenAI
from rich.console import Console

from agentbench.datasets.base import Dataset
from agentbench.proxy import ProxyServer
from agentbench.results import Results
from agentbench.types import Case, CaseResult, JudgeMetrics, Runner, RunnerContext, RunnerMetrics

console = Console()


class _AsyncCounter:
    """Async-safe counter for tracking progress."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = asyncio.Lock()

    async def increment(self) -> int:
        """Atomically increment and return new value."""
        async with self._lock:
            self._value += 1
            return self._value


class ProgressObserver:
    """Thread-safe observer for tracking and reporting case completion progress."""

    def __init__(self, total_cases: int, progress_counter: _AsyncCounter) -> None:
        """
        Initialize progress observer.

        Args:
            total_cases: Total number of cases to process
            progress_counter: Async-safe counter for tracking completed cases
        """
        self._total_cases = total_cases
        self._progress_counter = progress_counter
        self._console_lock = threading.Lock()

    async def on_case_complete(self, case_result: CaseResult, worker_id: int) -> None:  # noqa: PLR0912
        """
        Called when a case completes. Thread-safe reporting of case results.

        Args:
            case_result: Completed case result
            worker_id: ID of worker that processed the case
        """
        completed = await self._progress_counter.increment()

        llm_metrics = case_result.metrics.llm_metrics
        duration_s = case_result.metrics.model_duration_ms / 1000.0
        llm_call_count = llm_metrics.get("llm_call_count", 0)
        llm_input_tokens = llm_metrics.get("llm_input_tokens", 0)
        llm_output_tokens = llm_metrics.get("llm_output_tokens", 0)
        llm_cost_usd = llm_metrics.get("llm_total_cost_usd", 0.0)
        llm_latencies = llm_metrics.get("llm_latency_ms", [])
        avg_llm_latency_ms = sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0.0

        status_color = "green" if case_result.evaluation.passed else "red"
        status_icon = "✓" if case_result.evaluation.passed else "✗"
        status_text = "PASS" if case_result.evaluation.passed else "FAIL"
        console.print(
            f"[{status_color}]{status_icon}[/{status_color}] "
            f"[bold]{case_result.case_id}[/bold] "
            f"(worker {worker_id}, {completed}/{self._total_cases}) "
            f"[{status_color}]{status_text}[/{status_color}] "
            f"| {duration_s:.2f}s "
            f"| {llm_call_count} calls "
            f"| {llm_input_tokens + llm_output_tokens} tokens "
            f"| ${llm_cost_usd:.8f}"
        )

        if llm_latencies:
            sorted_latencies = sorted(llm_latencies)
            p50_latency = sorted_latencies[len(sorted_latencies) // 2]
            p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            console.print(
                f"  └─ LLM latency: {avg_llm_latency_ms:.0f}ms avg, "
                f"{p50_latency:.0f}ms p50, {p95_latency:.0f}ms p95 "
                f"({llm_input_tokens} in, {llm_output_tokens} out)"
            )

        if case_result.evaluation.f1_score is not None:
            f1_color = "green" if case_result.evaluation.f1_passed else "yellow"
            f1_status = "PASS" if case_result.evaluation.f1_passed else "FAIL"
            console.print(
                f"  └─ F1: [{f1_color}]{case_result.evaluation.f1_score:.3f} ({f1_status})[/{f1_color}]"
            )

        if case_result.evaluation.judge_passed is not None:
            judge_color = "green" if case_result.evaluation.judge_passed else "red"
            judge_status = "PASS" if case_result.evaluation.judge_passed else "FAIL"
            console.print(f"  └─ Judge: [{judge_color}]{judge_status}[/{judge_color}]")
            if case_result.judge_metrics:
                console.print(
                    f"      ({case_result.judge_metrics.input_tokens + case_result.judge_metrics.output_tokens} tokens)"
                )

        if case_result.error:
            console.print(f"  └─ [red]Error: {case_result.error}[/red]")


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
    proxy.metrics.clear_metrics("_current")
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


async def _worker(
    worker_id: int,
    case_queue: asyncio.Queue[Case | None],
    runner: Runner,
    proxies: list[ProxyServer],
    dataset: Dataset,
    model: str,
    openai_api_key: str,
    embedding_model: str | None,
    judge_model: str,
    judge_client: AsyncOpenAI,
    results: Results,
    observer: ProgressObserver,
) -> None:
    """
    Worker coroutine that processes cases from the queue.

    Each worker has a stable worker_id and uses its own proxy for metrics isolation.

    Args:
        worker_id: Stable worker identifier (0..concurrency-1)
        case_queue: Queue of cases to process (None signals end)
        runner: Runner instance
        proxies: List of proxy servers (one per worker)
        dataset: Dataset instance
        model: Model name
        openai_api_key: OpenAI API key
        embedding_model: Embedding model name if applicable
        judge_model: Judge model name
        judge_client: OpenAI client for judge
        results: Results container
        observer: Progress observer for reporting case completion
    """
    proxy = proxies[worker_id]

    while True:
        case = await case_queue.get()
        if case is None:
            case_queue.task_done()
            break

        runner_ctx = RunnerContext(
            model=model,
            proxy_port=proxy.port,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            case_id=case.id,
            worker_id=worker_id,
        )

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
        await observer.on_case_complete(case_result, worker_id)

        case_queue.task_done()


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
    concurrency: int = 1,
) -> Results:
    """
    Run evaluation pipeline on a dataset using the given runner.

    This is the main library entry point:
    Dataset → load cases → Runner → Dataset.eval → Results container.

    Args:
        dataset: Dataset to evaluate
        runner: Runner instance
        model: Model name
        openai_api_key: OpenAI API key
        run_id: Run identifier
        output_dir: Output directory for results
        config: Run configuration
        embedding_model: Embedding model name if applicable
        judge_model: Judge model name
        upstream_base_url: Upstream base URL for proxy
        limit: Limit number of cases
        concurrency: Number of concurrent case evaluations (default: 1)

    Returns:
        Results object containing all case results and metrics for this run.
    """
    console.print("[yellow]Loading dataset cases...[/yellow]")
    cases_iter = await dataset.load(limit=limit)
    cases = list(cases_iter)
    console.print(f"[green]✓ Loaded {len(cases)} cases[/green]")
    console.print()

    console.print(f"[yellow]Starting {concurrency} LLM proxy server(s)...[/yellow]")
    proxies: list[ProxyServer] = []
    for i in range(concurrency):
        proxy_port = portpicker.pick_unused_port()
        proxy = ProxyServer(
            port=proxy_port,
            openai_api_key=openai_api_key,
            upstream_base_url=upstream_base_url,
        )
        proxy.start()
        proxies.append(proxy)
        console.print(f"[green]✓ Proxy server {i} started on http://localhost:{proxy_port}[/green]")
    console.print()

    results = Results(output_dir=output_dir, config=config, run_id=run_id)
    completed_ids = results.get_completed_case_ids()

    if completed_ids:
        console.print(f"[cyan]Found {len(completed_ids)} completed cases in {output_dir}[/cyan]")

    judge_client = AsyncOpenAI()

    console.print("[yellow]Running evaluation...[/yellow]")
    if concurrency > 1:
        console.print(f"[yellow]Concurrency: {concurrency} workers[/yellow]")
    console.print()

    case_queue: asyncio.Queue[Case | None] = asyncio.Queue()

    cases_to_run = [c for c in cases if c.id not in completed_ids]
    skipped_count = len(cases) - len(cases_to_run)

    if skipped_count > 0:
        console.print(f"[cyan]Skipping {skipped_count} already completed cases[/cyan]")

    for case in cases_to_run:
        await case_queue.put(case)

    for _ in range(concurrency):
        await case_queue.put(None)

    progress_counter = _AsyncCounter()
    # If we are resuming, we want the progress to reflect total vs completed including past?
    # Observer takes 'total_cases'. Let's set it to the number of cases we are actually running
    # OR keep it as total dataset size and initialize counter?
    # Simplest is to show progress for *this run*.
    observer = ProgressObserver(total_cases=len(cases_to_run), progress_counter=progress_counter)


    start_time = time.time()

    workers = [
        _worker(
            worker_id=i,
            case_queue=case_queue,
            runner=runner,
            proxies=proxies,
            dataset=dataset,
            model=model,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            judge_model=judge_model,
            judge_client=judge_client,
            results=results,
            observer=observer,
        )
        for i in range(concurrency)
    ]

    await asyncio.gather(*workers)

    end_time = time.time()
    total_duration_s = end_time - start_time

    console.print()
    console.print(
        f"[green]✓ Evaluation complete[/green] "
        f"(total duration: {total_duration_s:.2f}s, "
        f"{total_duration_s / 60:.1f}min)"
    )
    console.print()

    for proxy in proxies:
        proxy.stop()
    results.finalize()

    return results
