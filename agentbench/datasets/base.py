from abc import ABC, abstractmethod
from pathlib import Path

from agentbench.config import DatasetConfig
from agentbench.context import EvalContext
from agentbench.types import Case


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
        candidate = Path(pattern).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        return str(self.root_dir / candidate)

    def _prepare_case(
        self,
        record: dict,
        fallback_id: str,
        input_key: str,
        expected_key: str,
    ) -> Case | None:
        case_input = record.get(input_key)
        if case_input is None:
            return None
        expected = record.get(expected_key)
        case_id = str(record.get("case_id", record.get("id", fallback_id)))
        history = record.get("history")
        history_list = history if isinstance(history, list) else []
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {input_key, expected_key, "history"}
        }
        return Case(
            case_id=case_id,
            dataset_name=self.config.name,
            input=str(case_input),
            expected=str(expected) if expected is not None else None,
            history=history_list,
            metadata=metadata,
        )
