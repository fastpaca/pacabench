"""Results container with incremental saving and rolling aggregated metrics."""

import json
from pathlib import Path
from typing import Any

from agentbench.metrics import AggregatedMetrics, aggregate_results
from agentbench.types import CaseResult


class Results:
    """Results container that accumulates cases in memory and saves to disk on finalize."""

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
        self.cases_map: dict[str, CaseResult] = {}  # Fast lookup by case_id
        self.metrics: AggregatedMetrics | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing if available
        self._load_existing_results()

        # Write config if not exists (or update? usually config is static for a run)
        if not (self.output_dir / "config.json").exists():
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(self.config, f, indent=2)

    def _load_existing_results(self) -> None:
        """Load existing results from jsonl file if it exists."""
        results_file = self.output_dir / "results.jsonl"
        if not results_file.exists():
            return

        try:
            with open(results_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    # We only need the case_id for filtering, but we load minimal info
                    # Note: Reconstructing full CaseResult might be complex if schema changes,
                    # but for filtering we primarily need the ID.
                    # For now, we'll store the ID in a set for fast checking.
                    case_id = data.get("case_id")
                    if case_id:
                        self.cases_map[case_id] = None  # Placeholder, we don't need full object yet
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")

    def add_case(self, case_result: CaseResult) -> None:
        """
        Add a case result to the collection and write to disk immediately.

        Args:
            case_result: Case result to add
        """
        self.cases.append(case_result)
        self.cases_map[case_result.case_id] = case_result

        # Append to jsonl immediately
        with open(self.output_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(_case_result_to_dict(case_result)) + "\n")

    def get_completed_case_ids(self) -> set[str]:
        """Get set of case IDs that have already been processed."""
        return set(self.cases_map.keys())

    def finalize(self) -> None:
        """Compute final aggregated metrics from all results on disk."""
        all_cases = []
        results_file = self.output_dir / "results.jsonl"

        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Try to validate as CaseResult
                        # Handle legacy flat format if needed, or just proceed with current format
                        # We are switching to model_dump(), so new lines are compatible.
                        # Old lines (flat) won't validate. We could migrate them or ignore.
                        # Given this is a dev tool, ignoring legacy incompatible lines is acceptable for now.
                        case_result = CaseResult.model_validate(data)
                        all_cases.append(case_result)
                    except Exception:
                        # Fallback: if we can't validate, maybe it's the old flat format.
                        # We could try to reconstruct, but for now let's just skip aggregation of legacy lines.
                        continue

        # If we couldn't load anything (e.g. all legacy), fall back to in-memory cases
        if not all_cases:
            all_cases = self.cases

        self.metrics = aggregate_results(all_cases)

        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics.model_dump(exclude_none=False), f, indent=2)


def _case_result_to_dict(result: CaseResult) -> dict[str, Any]:
    """Convert CaseResult to dict for JSON serialization."""
    return result.model_dump(mode="json")
