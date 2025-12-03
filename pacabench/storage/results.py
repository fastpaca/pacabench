"""Results loading and metrics calculation."""

import math
from pathlib import Path
from typing import overload

import orjson

from pacabench.models import AggregatedMetrics, CaseResult


@overload
def load_results(run_dir: Path, *, raw: bool = False) -> list[CaseResult]: ...


@overload
def load_results(run_dir: Path, *, raw: bool = True) -> list[dict]: ...


def load_results(run_dir: Path, *, raw: bool = False) -> list[CaseResult] | list[dict]:
    """Load results from a run directory.

    Args:
        run_dir: Path to the run directory.
        raw: If True, return raw dicts (faster). If False, return CaseResult models.

    Returns:
        List of results, deduplicated by (agent, dataset, case_id).
    """
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return []

    deduped: dict[tuple[str | None, str | None, str | None], dict] = {}
    with open(results_path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = orjson.loads(line)
                key = (data.get("agent_name"), data.get("dataset_name"), data.get("case_id"))
                deduped[key] = data
            except orjson.JSONDecodeError:
                pass

    if raw:
        return list(deduped.values())
    return [CaseResult.model_validate(data) for data in deduped.values()]


def load_results_raw(run_dir: Path) -> list[dict]:
    """Load results as raw dicts. Alias for load_results(raw=True)."""
    return load_results(run_dir, raw=True)


def _calculate_percentile(data: list[float], percentile: float) -> float:
    """Calculate percentile using linear interpolation (numpy-style)."""
    if not data:
        return 0.0
    data.sort()
    index = (len(data) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return data[int(index)]
    return data[lower] * (upper - index) + data[upper] * (index - lower)


def calculate_metrics(results: list[dict] | list[CaseResult]) -> AggregatedMetrics:
    """Calculate aggregated metrics from results.

    Accepts either raw dicts or CaseResult models for flexibility.
    """
    total = len(results)
    if total == 0:
        return AggregatedMetrics()

    def _get(r: dict | CaseResult, key: str, default: float = 0.0) -> float:
        if isinstance(r, dict):
            return r.get(key, default) or default
        return getattr(r, key, default) or default

    def _get_dict(r: dict | CaseResult, key: str) -> dict:
        if isinstance(r, dict):
            return r.get(key, {}) or {}
        return getattr(r, key, {}) or {}

    passed = sum(1 for r in results if _get(r, "passed"))
    accuracy = passed / total

    durations = [_get(r, "runner_duration_ms") for r in results]
    p50_dur = _calculate_percentile(durations, 0.50)
    p95_dur = _calculate_percentile(durations, 0.95)

    latencies: list[float] = []
    total_cost = 0.0
    total_judge_cost = 0.0
    total_input = 0
    total_output = 0

    for r in results:
        metrics = _get_dict(r, "llm_metrics")
        total_cost += metrics.get("llm_total_cost_usd", 0.0) or 0.0
        total_input += metrics.get("llm_input_tokens", 0) or 0
        total_output += metrics.get("llm_output_tokens", 0) or 0
        total_judge_cost += _get(r, "judge_cost_usd")

        lats = metrics.get("llm_latency_ms", [])
        if isinstance(lats, list):
            latencies.extend(lats)
        elif isinstance(lats, (int, float)):
            latencies.append(float(lats))

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p50_lat = _calculate_percentile(latencies, 0.50)
    p95_lat = _calculate_percentile(latencies, 0.95)

    return AggregatedMetrics(
        accuracy=accuracy,
        total_cases=total,
        failed_cases=total - passed,
        p50_duration_ms=p50_dur,
        p95_duration_ms=p95_dur,
        avg_llm_latency_ms=avg_lat,
        p50_llm_latency_ms=p50_lat,
        p95_llm_latency_ms=p95_lat,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=total_cost,
        total_judge_cost_usd=total_judge_cost,
    )
