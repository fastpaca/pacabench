"""Results container with incremental saving and rolling aggregated metrics."""

import json
from pathlib import Path
from typing import Any

from agentbench.metrics import AggregatedMetrics, aggregate_results
from agentbench.types import CaseResult


class Results:
    """Results container that ingests cases incrementally and saves to disk."""

    def __init__(self, output_dir: Path, config: dict[str, Any], run_id: str) -> None:
        """
        Initialize results container.

        Args:
            output_dir: Directory to save results
            config: Run configuration
            run_id: Run identifier
        """
        self.output_dir = output_dir
        self.config = config
        self.run_id = run_id
        self.cases: list[CaseResult] = []
        self.metrics: AggregatedMetrics | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        results_file = self.output_dir / "results.jsonl"
        if results_file.exists():
            results_file.unlink()

    def add_case(self, case_result: CaseResult) -> None:
        """
        Add a case result and update metrics/files incrementally.

        Args:
            case_result: Case result to add
        """
        self.cases.append(case_result)

        with open(self.output_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(_case_result_to_dict(case_result)) + "\n")
        # Aggregated metrics are computed on finalize() to avoid recomputing on every case.

    def finalize(self) -> None:
        """Compute final aggregated metrics and write metrics to disk."""
        self.metrics = aggregate_results(self.cases)
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics.model_dump(exclude_none=False), f, indent=2)


def _case_result_to_dict(result: CaseResult) -> dict[str, Any]:
    """Convert CaseResult to dict for JSON serialization."""
    judge_metrics_dict = None
    if result.judge_metrics:
        judge_metrics_dict = {
            "input_tokens": result.judge_metrics.input_tokens,
            "output_tokens": result.judge_metrics.output_tokens,
        }
    return {
        "case_id": result.case_id,
        "passed": result.evaluation.passed,
        "output": result.output,
        "error": result.error,
        "model_duration_ms": result.metrics.model_duration_ms,
        "llm_metrics": result.metrics.llm_metrics,
        "f1_score": result.evaluation.f1_score,
        "f1_passed": result.evaluation.f1_passed,
        "judge_passed": result.evaluation.judge_passed,
        "judge_metrics": judge_metrics_dict,
    }
