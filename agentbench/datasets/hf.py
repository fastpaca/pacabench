import logging

from datasets import load_dataset as hf_load_dataset

from agentbench.datasets.base import BaseDataset
from agentbench.types import Case

logger = logging.getLogger(__name__)


class HuggingFaceDataset(BaseDataset):
    def load(self, limit: int | None = None) -> list[Case]:
        source = self.config.source
        dataset_name = (
            source[len("huggingface:") :] if source.startswith("huggingface:") else source
        )

        split = self.config.split or "train"

        # HF load_dataset doesn't support limit easily at load time without streaming=True
        # But for simplicity, let's load and slice.
        # Or use split slicing string: "train[:limit]"

        split_str = f"{split}[:{limit}]" if limit is not None else split

        logger.info(f"Loading HF dataset: {dataset_name} split={split_str}")
        ds = hf_load_dataset(dataset_name, split=split_str)

        cases = []
        input_key = self.config.input_map.get("input", "input")
        expected_key = self.config.input_map.get("expected", "expected")

        for i, item in enumerate(ds):
            case_input = item.get(input_key)
            if case_input is None:
                continue

            case_expected = item.get(expected_key)
            c_id = str(item.get("case_id", item.get("id", str(i))))

            cases.append(
                Case(
                    case_id=c_id,
                    dataset_name=self.config.name,
                    input=str(case_input),
                    expected=str(case_expected) if case_expected is not None else None,
                    metadata=dict(item),
                )
            )

        return cases
