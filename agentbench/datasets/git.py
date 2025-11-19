import glob
import json
import logging
import os
import subprocess
from pathlib import Path

from git import Repo

from agentbench.datasets.base import BaseDataset
from agentbench.types import Case

logger = logging.getLogger(__name__)


class GitDataset(BaseDataset):
    def load(self, limit: int | None = None) -> list[Case]:
        source = self.config.source
        repo_url = source[len("git:") :] if source.startswith("git:") else source

        repo_name = repo_url.split("/")[-1].replace(".git", "")
        cache_dir = Path.home() / ".cache" / "agentbench" / "repos" / repo_name

        self._ensure_repo(repo_url, cache_dir)

        if self.config.prepare:
            logger.info(f"Running prepare script: {self.config.prepare}")
            env = os.environ.copy()
            env["AGENTBENCH_DATASET_PATH"] = str(cache_dir)
            subprocess.check_call(self.config.prepare, shell=True, env=env)

        files = glob.glob(str(cache_dir / "**/*.jsonl"), recursive=True)

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

    def _ensure_repo(self, url: str, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (dest / ".git").exists():
            try:
                repo = Repo(dest)
                repo.remotes.origin.pull()
            except Exception as e:
                logger.warning(f"Failed to update repo at {dest}: {e}")
        else:
            Repo.clone_from(url, dest)
