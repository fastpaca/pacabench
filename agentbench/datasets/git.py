import glob
import json
import logging
import subprocess
from pathlib import Path

from git import Repo

from agentbench.datasets.base import BaseDataset
from agentbench.types import Case

logger = logging.getLogger(__name__)


class GitDataset(BaseDataset):
    def load(self, limit: int | None = None) -> list[Case]:
        repo_url = self._normalize_repo_url(self.config.source)
        repo_dir = self._ensure_repo(repo_url)

        if self.config.prepare:
            self._run_prepare(repo_dir)

        files = glob.glob(str(repo_dir / "**" / "*.jsonl"), recursive=True)
        input_key = self.config.input_map.get("input", "input")
        expected_key = self.config.input_map.get("expected", "expected")

        cases: list[Case] = []
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

    def _normalize_repo_url(self, source: str) -> str:
        return source[len("git:") :] if source.startswith("git:") else source

    def _repo_cache_dir(self, repo_url: str) -> Path:
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        return self.datasets_cache_dir / "repos" / repo_name

    def _ensure_repo(self, repo_url: str) -> Path:
        repo_dir = self._repo_cache_dir(repo_url)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if repo_dir.exists() and (repo_dir / ".git").exists():
            try:
                repo = Repo(repo_dir)
                repo.remotes.origin.pull()
            except Exception as exc:
                logger.warning("Failed to update repo at %s: %s", repo_dir, exc)
        else:
            Repo.clone_from(repo_url, repo_dir)
        return repo_dir

    def _run_prepare(self, repo_dir: Path) -> None:
        logger.info("Running prepare script for %s: %s", self.config.name, self.config.prepare)
        env = self.ctx.env.copy()
        env["AGENTBENCH_DATASET_PATH"] = str(repo_dir)
        subprocess.run(
            self.config.prepare,
            shell=True,
            cwd=repo_dir,
            check=True,
            env=env,
        )
