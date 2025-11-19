import logging

from datasets import load_dataset as hf_load_dataset

from agentbench.datasets.base import BaseDataset
from agentbench.types import Case

logger = logging.getLogger(__name__)


class HuggingFaceDataset(BaseDataset):
    def load(self, limit: int | None = None) -> list[Case]:
        dataset_name = self._normalize_name(self.config.source)
        split = self.config.split or "train"
        split_str = f"{split}[:{limit}]" if limit is not None else split

        logger.info("Loading HF dataset %s split=%s", dataset_name, split_str)
        ds = hf_load_dataset(dataset_name, split=split_str)

        cases: list[Case] = []
        input_key = self.config.input_map.get("input", "input")
        expected_key = self.config.input_map.get("expected", "expected")

        for i, item in enumerate(ds):
            record = dict(item)
            case = self._prepare_case(
                record=record,
                fallback_id=str(i),
                input_key=input_key,
                expected_key=expected_key,
            )
            if case is None:
                continue
            cases.append(case)

        return cases

    def _normalize_name(self, source: str) -> str:
        return source[len("huggingface:") :] if source.startswith("huggingface:") else source
