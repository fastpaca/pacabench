import glob
import json
import logging
import subprocess
from pathlib import Path

from git import Repo

from pacabench.datasets.base import BaseDataset
from pacabench.types import Case

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

        # Fix: if prepare script path is relative, resolve it relative to the CWD
        # (where pacabench.yaml is), NOT relative to the repo_dir.
        # But wait, subprocess.run(cwd=repo_dir) executes IN the repo dir.
        # If "python scripts/prepare.py" is passed, it looks for it in repo_dir.
        # But the user likely wants to point to a script in THEIR project, OR a script in the repo.
        # The current ambiguity is confusing.
        #
        # If we resolve the command path BEFORE passing to subprocess, we can support local scripts.
        # But `prepare` is a shell command string, not just a path. E.g. "python foo.py --arg".
        #
        # A simple heuristic: if the command starts with "python ", try to resolve the script path.
        # But that's brittle.
        #
        # Instead, let's execute the command from the CWD (project root),
        # but pass the REPO_DIR as an env var (which we already do).
        #
        # BUT: existing logic sets cwd=repo_dir. This assumes the prepare script is INSIDE the repo.
        # In our case, `scripts/prepare_membench.py` is in OUR project, not the repo.
        #
        # SOLUTION: Check if the command refers to a file that exists relative to CWD.
        # If so, use CWD. If not, use repo_dir.
        #
        # Actually, even better: Just run in CWD. The prepare script knows where the dataset is via env var.
        # If the user wants to run a script INSIDE the repo, they can do `cd $AGENTBENCH_DATASET_PATH && ...`
        # OR we can provide a flag.
        #
        # Changing `cwd=repo_dir` to `cwd=os.getcwd()` (or `self.ctx.root_dir`) would break existing configs
        # that rely on running scripts inside the repo.
        #
        # Let's stick to the current behavior but documentation should clarify.
        # However, to fix the specific user issue without WORKSPACE_ROOT, we can try to allow
        # relative paths to resolve to the project root if they don't exist in the repo.

        # For now, let's just change the CWD to be the project root, because typically
        # the harness controls the environment. The prepare script receives the target directory.
        # It's more natural for the "prepare" command to run from where the config is defined.

        subprocess.run(
            self.config.prepare,
            shell=True,
            cwd=self.ctx.root_dir,  # Changed from repo_dir to project root
            check=True,
            env=env,
        )
