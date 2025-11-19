import glob
import json
from pathlib import Path

from agentbench.datasets.base import BaseDataset
from agentbench.types import Case


class LocalDataset(BaseDataset):
    def load(self, limit: int | None = None) -> list[Case]:
        source_path = self.config.source
        if "*" in source_path:
            files = glob.glob(source_path, recursive=True)
        else:
            p = Path(source_path)
            files = glob.glob(str(p / "*.jsonl"), recursive=True) if p.is_dir() else [source_path]

        cases = []
        input_key = self.config.input_map.get("input", "input")
        expected_key = self.config.input_map.get("expected", "expected")

        count = 0
        for fpath in files:
            if limit is not None and count >= limit:
                break
            with open(fpath) as f:
                for i, line in enumerate(f):
                    if limit is not None and count >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    case_input = data.get(input_key)
                    if case_input is None:
                        continue

                    case_expected = data.get(expected_key)
                    c_id = str(data.get("case_id", data.get("id", f"{Path(fpath).stem}-{i}")))

                    cases.append(
                        Case(
                            case_id=c_id,
                            dataset_name=self.config.name,
                            input=str(case_input),
                            expected=str(case_expected) if case_expected is not None else None,
                            metadata=data,
                        )
                    )
                    count += 1
        return cases
