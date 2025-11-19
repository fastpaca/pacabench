import glob
import json
from pathlib import Path

from agentbench.datasets.base import BaseDataset
from agentbench.types import Case


class LocalDataset(BaseDataset):
    def load(self, limit: int | None = None) -> list[Case]:
        source_path = self.config.source
        if "*" in source_path:
            pattern = self.resolve_pattern(source_path)
            files = glob.glob(pattern, recursive=True)
        else:
            p = self.resolve_path(source_path)
            files = glob.glob(str(p / "*.jsonl"), recursive=True) if p.is_dir() else [str(p)]

        cases: list[Case] = []
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
                    case = self._prepare_case(
                        record=data,
                        fallback_id=f"{Path(fpath).stem}-{i}",
                        input_key=input_key,
                        expected_key=expected_key,
                    )
                    if case is None:
                        continue
                    cases.append(case)
                    count += 1
        return cases
