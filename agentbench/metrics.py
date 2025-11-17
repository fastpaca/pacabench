"""Aggregated run-level metrics and helpers."""

import numpy as np
from pydantic import BaseModel, Field

from agentbench.types import CaseResult


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across all cases in a run."""

    total_cases: int = Field(..., description="Total number of cases")
    passed_cases: int = Field(..., description="Number of passed cases")
    failed_cases: int = Field(..., description="Number of failed cases")
    accuracy: float = Field(..., description="Overall accuracy")
    precision: float = Field(..., description="Overall precision")
    total_duration_s: float = Field(..., description="Total duration in seconds")
    avg_duration_s: float = Field(..., description="Average duration per case in seconds")
    p50_duration_s: float = Field(..., description="P50 duration in seconds")
    p95_duration_s: float = Field(..., description="P95 duration in seconds")
    p99_duration_s: float = Field(..., description="P99 duration in seconds")
    total_llm_calls: int = Field(..., description="Total LLM API calls")
    total_input_tokens: int = Field(..., description="Total input tokens")
    total_output_tokens: int = Field(..., description="Total output tokens")
    total_cache_read_tokens: int = Field(..., description="Total cache read tokens")
    total_cache_write_tokens: int = Field(..., description="Total cache write tokens")
    avg_input_tokens: float = Field(..., description="Average input tokens per case")
    avg_output_tokens: float = Field(..., description="Average output tokens per case")
    avg_llm_latency_ms: float = Field(..., description="Average LLM latency in milliseconds")
    p50_llm_latency_ms: float = Field(..., description="P50 LLM latency in milliseconds")
    p95_llm_latency_ms: float = Field(..., description="P95 LLM latency in milliseconds")
    p99_llm_latency_ms: float = Field(..., description="P99 LLM latency in milliseconds")
    total_cost_usd: float = Field(..., description="Total cost in USD")
    avg_cost_per_case_usd: float = Field(..., description="Average cost per case in USD")
    avg_f1_score: float | None = Field(None, description="Average F1 score")
    total_judge_input_tokens: int | None = Field(None, description="Total judge input tokens")
    total_judge_output_tokens: int | None = Field(None, description="Total judge output tokens")

    model_config = {"extra": "forbid"}

    @classmethod
    def from_results(cls, cases: list[CaseResult]) -> "AggregatedMetrics":
        """Create aggregated metrics from a list of case results."""
        if not cases:
            return _empty_metrics()

        total_cases = len(cases)
        passed_cases = sum(1 for r in cases if r.evaluation.passed)
        failed_cases = total_cases - passed_cases
        accuracy = passed_cases / total_cases if total_cases > 0 else 0.0

        evaluator_passes = []
        for r in cases:
            case_evaluator_count = 0
            case_evaluator_passes = 0

            if r.evaluation.f1_passed is not None:
                case_evaluator_count += 1
                if r.evaluation.f1_passed:
                    case_evaluator_passes += 1

            if r.evaluation.judge_passed is not None:
                case_evaluator_count += 1
                if r.evaluation.judge_passed:
                    case_evaluator_passes += 1

            if case_evaluator_count > 0:
                evaluator_passes.append(case_evaluator_passes / case_evaluator_count)
            else:
                evaluator_passes.append(1.0 if r.evaluation.passed else 0.0)

        precision = float(np.mean(evaluator_passes)) if evaluator_passes else 0.0

        durations = [r.metrics.model_duration_ms / 1000 for r in cases]
        total_duration_s = sum(durations)
        avg_duration_s = np.mean(durations) if durations else 0.0
        p50_duration_s = float(np.percentile(durations, 50)) if durations else 0.0
        p95_duration_s = float(np.percentile(durations, 95)) if durations else 0.0
        p99_duration_s = float(np.percentile(durations, 99)) if durations else 0.0

        total_llm_calls = sum(r.metrics.llm_metrics.get("llm_call_count", 0) for r in cases)
        total_input_tokens = sum(r.metrics.llm_metrics.get("llm_input_tokens", 0) for r in cases)
        total_output_tokens = sum(r.metrics.llm_metrics.get("llm_output_tokens", 0) for r in cases)
        total_cache_read_tokens = sum(
            r.metrics.llm_metrics.get("llm_cache_read_tokens", 0) for r in cases
        )
        total_cache_write_tokens = sum(
            r.metrics.llm_metrics.get("llm_cache_write_tokens", 0) for r in cases
        )
        total_cost_usd = sum(r.metrics.llm_metrics.get("llm_total_cost_usd", 0.0) for r in cases)

        avg_input_tokens = total_input_tokens / total_cases if total_cases > 0 else 0.0
        avg_output_tokens = total_output_tokens / total_cases if total_cases > 0 else 0.0
        avg_cost_per_case_usd = total_cost_usd / total_cases if total_cases > 0 else 0.0

        all_latencies = []
        for r in cases:
            latencies = r.metrics.llm_metrics.get("llm_latency_ms", [])
            if isinstance(latencies, list):
                all_latencies.extend(latencies)

        avg_llm_latency_ms = float(np.mean(all_latencies)) if all_latencies else 0.0
        p50_llm_latency_ms = float(np.percentile(all_latencies, 50)) if all_latencies else 0.0
        p95_llm_latency_ms = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
        p99_llm_latency_ms = float(np.percentile(all_latencies, 99)) if all_latencies else 0.0

        f1_scores = [r.evaluation.f1_score for r in cases if r.evaluation.f1_score is not None]
        avg_f1_score = float(np.mean(f1_scores)) if f1_scores else None

        judge_input_tokens = [r.judge_metrics.input_tokens for r in cases if r.judge_metrics]
        judge_output_tokens = [r.judge_metrics.output_tokens for r in cases if r.judge_metrics]
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


def aggregate_results(cases: list[CaseResult]) -> AggregatedMetrics:
    """Aggregate metrics for a list of case results."""
    return AggregatedMetrics.from_results(cases)


def _empty_metrics() -> AggregatedMetrics:
    """Return empty aggregated metrics."""
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
