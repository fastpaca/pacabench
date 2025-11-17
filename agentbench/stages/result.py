"""Stage 4: Results - Metrics collection and aggregation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agentbench.context import EvalContext


@dataclass
class CaseResult:
    """Result for a single test case."""

    case_id: str
    passed: bool
    output: str | None
    error: str | None
    runner_duration_ms: float
    llm_metrics: dict[str, Any]
    f1_score: float | None = None
    f1_passed: bool | None = None
    judge_passed: bool | None = None
    judge_metrics: dict[str, int] | None = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all cases."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    accuracy: float
    precision: float
    total_duration_s: float
    avg_duration_s: float
    p50_duration_s: float
    p95_duration_s: float
    p99_duration_s: float
    total_llm_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_llm_latency_ms: float
    p50_llm_latency_ms: float
    p95_llm_latency_ms: float
    p99_llm_latency_ms: float
    total_cost_usd: float
    avg_cost_per_case_usd: float
    avg_f1_score: float | None = None
    total_judge_input_tokens: int | None = None
    total_judge_output_tokens: int | None = None


def aggregate_results(results: list[CaseResult]) -> AggregatedMetrics:
    """
    Aggregate results from multiple cases.

    Args:
        results: List of case results

    Returns:
        Aggregated metrics
    """
    if not results:
        return _empty_metrics()

    total_cases = len(results)
    passed_cases = sum(1 for r in results if r.passed)
    failed_cases = total_cases - passed_cases
    accuracy = passed_cases / total_cases if total_cases > 0 else 0.0

    evaluator_passes = []
    for r in results:
        case_evaluator_count = 0
        case_evaluator_passes = 0

        if r.f1_passed is not None:
            case_evaluator_count += 1
            if r.f1_passed:
                case_evaluator_passes += 1

        if r.judge_passed is not None:
            case_evaluator_count += 1
            if r.judge_passed:
                case_evaluator_passes += 1

        if case_evaluator_count > 0:
            evaluator_passes.append(case_evaluator_passes / case_evaluator_count)
        else:
            evaluator_passes.append(1.0 if r.passed else 0.0)

    precision = float(np.mean(evaluator_passes)) if evaluator_passes else 0.0

    durations = [r.runner_duration_ms / 1000 for r in results]
    total_duration_s = sum(durations)
    avg_duration_s = np.mean(durations) if durations else 0.0
    p50_duration_s = float(np.percentile(durations, 50)) if durations else 0.0
    p95_duration_s = float(np.percentile(durations, 95)) if durations else 0.0
    p99_duration_s = float(np.percentile(durations, 99)) if durations else 0.0

    total_llm_calls = sum(r.llm_metrics.get("llm_call_count", 0) for r in results)
    total_input_tokens = sum(r.llm_metrics.get("llm_input_tokens", 0) for r in results)
    total_output_tokens = sum(r.llm_metrics.get("llm_output_tokens", 0) for r in results)
    total_cache_read_tokens = sum(r.llm_metrics.get("llm_cache_read_tokens", 0) for r in results)
    total_cache_write_tokens = sum(r.llm_metrics.get("llm_cache_write_tokens", 0) for r in results)
    total_cost_usd = sum(r.llm_metrics.get("llm_total_cost_usd", 0.0) for r in results)

    avg_input_tokens = total_input_tokens / total_cases if total_cases > 0 else 0.0
    avg_output_tokens = total_output_tokens / total_cases if total_cases > 0 else 0.0
    avg_cost_per_case_usd = total_cost_usd / total_cases if total_cases > 0 else 0.0

    all_latencies = []
    for r in results:
        latencies = r.llm_metrics.get("llm_latency_ms", [])
        if isinstance(latencies, list):
            all_latencies.extend(latencies)

    avg_llm_latency_ms = float(np.mean(all_latencies)) if all_latencies else 0.0
    p50_llm_latency_ms = float(np.percentile(all_latencies, 50)) if all_latencies else 0.0
    p95_llm_latency_ms = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    p99_llm_latency_ms = float(np.percentile(all_latencies, 99)) if all_latencies else 0.0

    f1_scores = [r.f1_score for r in results if r.f1_score is not None]
    avg_f1_score = float(np.mean(f1_scores)) if f1_scores else None

    judge_input_tokens = [
        r.judge_metrics.get("input_tokens", 0) for r in results if r.judge_metrics
    ]
    judge_output_tokens = [
        r.judge_metrics.get("output_tokens", 0) for r in results if r.judge_metrics
    ]
    total_judge_input_tokens = sum(judge_input_tokens) if judge_input_tokens else None
    total_judge_output_tokens = sum(judge_output_tokens) if judge_output_tokens else None

    return AggregatedMetrics(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        accuracy=accuracy,
        precision=precision,
        total_duration_s=total_duration_s,
        avg_duration_s=avg_duration_s,
        p50_duration_s=p50_duration_s,
        p95_duration_s=p95_duration_s,
        p99_duration_s=p99_duration_s,
        total_llm_calls=total_llm_calls,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cache_read_tokens=total_cache_read_tokens,
        total_cache_write_tokens=total_cache_write_tokens,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        avg_llm_latency_ms=avg_llm_latency_ms,
        p50_llm_latency_ms=p50_llm_latency_ms,
        p95_llm_latency_ms=p95_llm_latency_ms,
        p99_llm_latency_ms=p99_llm_latency_ms,
        total_cost_usd=total_cost_usd,
        avg_cost_per_case_usd=avg_cost_per_case_usd,
        avg_f1_score=avg_f1_score,
        total_judge_input_tokens=total_judge_input_tokens,
        total_judge_output_tokens=total_judge_output_tokens,
    )


def save_results(
    output_dir: Path,
    results: list[CaseResult],
    metrics: AggregatedMetrics,
    config: dict[str, Any],
) -> None:
    """
    Save results to output directory.

    Args:
        output_dir: Directory to save results
        results: List of case results
        metrics: Aggregated metrics
        config: Run configuration
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(_result_to_dict(result)) + "\n")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(_metrics_to_dict(metrics), f, indent=2)


def _result_to_dict(result: CaseResult) -> dict[str, Any]:
    """Convert CaseResult to dict for JSON serialization."""
    return {
        "case_id": result.case_id,
        "passed": result.passed,
        "output": result.output,
        "error": result.error,
        "runner_duration_ms": result.runner_duration_ms,
        "llm_metrics": result.llm_metrics,
        "f1_score": result.f1_score,
        "f1_passed": result.f1_passed,
        "judge_passed": result.judge_passed,
        "judge_metrics": result.judge_metrics,
    }


def _metrics_to_dict(metrics: AggregatedMetrics) -> dict[str, Any]:
    """Convert AggregatedMetrics to dict for JSON serialization."""
    return {
        "total_cases": metrics.total_cases,
        "passed_cases": metrics.passed_cases,
        "failed_cases": metrics.failed_cases,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "total_duration_s": metrics.total_duration_s,
        "avg_duration_s": metrics.avg_duration_s,
        "p50_duration_s": metrics.p50_duration_s,
        "p95_duration_s": metrics.p95_duration_s,
        "p99_duration_s": metrics.p99_duration_s,
        "total_llm_calls": metrics.total_llm_calls,
        "total_input_tokens": metrics.total_input_tokens,
        "total_output_tokens": metrics.total_output_tokens,
        "total_cache_read_tokens": metrics.total_cache_read_tokens,
        "total_cache_write_tokens": metrics.total_cache_write_tokens,
        "avg_input_tokens": metrics.avg_input_tokens,
        "avg_output_tokens": metrics.avg_output_tokens,
        "avg_llm_latency_ms": metrics.avg_llm_latency_ms,
        "p50_llm_latency_ms": metrics.p50_llm_latency_ms,
        "p95_llm_latency_ms": metrics.p95_llm_latency_ms,
        "p99_llm_latency_ms": metrics.p99_llm_latency_ms,
        "total_cost_usd": metrics.total_cost_usd,
        "avg_cost_per_case_usd": metrics.avg_cost_per_case_usd,
        "avg_f1_score": metrics.avg_f1_score,
        "total_judge_input_tokens": metrics.total_judge_input_tokens,
        "total_judge_output_tokens": metrics.total_judge_output_tokens,
    }


def _empty_metrics() -> AggregatedMetrics:
    """Return empty metrics."""
    return AggregatedMetrics(
        total_cases=0,
        passed_cases=0,
        failed_cases=0,
        accuracy=0.0,
        precision=0.0,
        total_duration_s=0.0,
        avg_duration_s=0.0,
        p50_duration_s=0.0,
        p95_duration_s=0.0,
        p99_duration_s=0.0,
        total_llm_calls=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cache_read_tokens=0,
        total_cache_write_tokens=0,
        avg_input_tokens=0.0,
        avg_output_tokens=0.0,
        avg_llm_latency_ms=0.0,
        p50_llm_latency_ms=0.0,
        p95_llm_latency_ms=0.0,
        p99_llm_latency_ms=0.0,
        total_cost_usd=0.0,
        avg_cost_per_case_usd=0.0,
    )


def collect_metrics(ctx: EvalContext) -> dict[str, Any]:
    """Collect LLM metrics from proxy."""
    metrics = ctx.proxy.metrics.get_metrics("_current")
    ctx.proxy.metrics.clear_metrics("_current")
    return metrics
