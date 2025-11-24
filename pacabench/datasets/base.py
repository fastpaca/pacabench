from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pacabench.config import DatasetConfig
from pacabench.context import EvalContext
from pacabench.types import Case

# Common fields to exclude from metadata to prevent leakage
_COMMON_EXCLUDE_KEYS = {
    "history",
    "case_id",
    "id",
    "ground_truth",
    "answer",
    "solution",
    "explanation",
    "reasoning",
    "correct_answer",
    "label",
    "target",
}


class BaseDataset(ABC):
    def __init__(self, config: DatasetConfig, ctx: EvalContext):
        self.config = config
        self.ctx = ctx
        self.root_dir = ctx.root_dir
        self.datasets_cache_dir = ctx.datasets_cache_dir

    @abstractmethod
    def load(self, limit: int | None = None) -> list[Case]:
        """Load cases for this dataset, optionally limited."""
        pass

    def resolve_path(self, path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if path.is_absolute():
            return path
        return (self.root_dir / path).resolve()

    def resolve_pattern(self, pattern: str) -> str:
        # Helper for glob patterns which need string paths
        return str(self.resolve_path(pattern))

    def _prepare_case(
        self,
        record: dict[str, Any],
        fallback_id: str,
        input_key: str,
        expected_key: str,
    ) -> Case | None:
        case_input = record.get(input_key)
        if case_input is None:
            return None

        expected = record.get(expected_key)
        # Prioritize case_id, then id, then fallback
        case_id = str(record.get("case_id") or record.get("id") or fallback_id)

        history = record.get("history", [])
        if not isinstance(history, list):
            # Fallback or error? We'll coerce to empty list to be safe but strictly typed
            history = []

        # Combine dynamic keys with static exclude list
        exclude_keys = _COMMON_EXCLUDE_KEYS | {input_key, expected_key}
        metadata = {k: v for k, v in record.items() if k not in exclude_keys}

        return Case(
            case_id=case_id,
            dataset_name=self.config.name,
            input=str(case_input),
            expected=str(expected) if expected is not None else None,
            history=history,
            metadata=metadata,
        )
