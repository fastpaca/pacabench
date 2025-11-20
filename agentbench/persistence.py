import contextlib
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agentbench.config import BenchmarkConfig
from agentbench.context import EvalContext
from agentbench.types import CaseResult, ErrorType


class RunManager:
    def __init__(self, ctx: EvalContext, run_id: str | None = None, force_new_run: bool = False):
        self.ctx = ctx
        self.config = ctx.runtime_config
        self.base_config = ctx.base_config
        self.base_dir = ctx.runs_dir
        self._config_fingerprint = _compute_config_fingerprint(self.config)
        self.resuming = False
        self._force_new_run = force_new_run

        if run_id:
            self.run_dir = self._resolve_run_dir(run_id)
            self.run_dir.parent.mkdir(parents=True, exist_ok=True)
            self.run_id = run_id
            metadata = self._read_metadata_file(self.run_dir / "metadata.json")
            if self.run_dir.exists():
                fingerprint = metadata.get("config_fingerprint")
                if fingerprint and fingerprint == self._config_fingerprint:
                    if force_new_run:
                        raise ValueError(
                            f"Run directory {self.run_dir} already exists; cannot force a fresh run."
                        )
                    self.resuming = True
                elif not force_new_run:
                    raise ValueError(
                        f"Run directory {self.run_dir} exists but cannot be resumed. "
                        "Use --fresh-run or choose a different run id."
                    )
            else:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                self.resuming = False
        else:
            resume_dir = None if force_new_run else self._find_incomplete_run()
            if resume_dir:
                self.run_dir = resume_dir
                self.run_id = resume_dir.name
                self.resuming = True
            else:
                self.run_id = self._generate_run_id()
                self.run_dir = self.base_dir / self.run_id

        self.results_path = self.run_dir / "results.jsonl"
        self.errors_path = self.run_dir / "system_errors.jsonl"
        self.metadata_path = self.run_dir / "metadata.json"
        self.config_path = self.run_dir / "agentbench.yaml"
        self._completed_entries: set[tuple[str, str, str]] = set()
        self._case_attempts: dict[tuple[str, str, str], int] = defaultdict(int)
        self._load_completed_entries()
        self.completed_cases = len(self._completed_entries)
        self.total_cases = self._load_total_cases()
        self.system_error_count = self._load_system_error_count()

    def _generate_run_id(self) -> str:
        # format: dataset-agent-timestamp
        # But config has multiple datasets/agents.
        # Use config name + timestamp
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.config.name}-{ts}"

    def _resolve_run_dir(self, run_id: str) -> Path:
        candidate = Path(run_id)
        if candidate.is_absolute():
            return candidate
        return self.base_dir / run_id

    def _load_total_cases(self) -> int:
        metadata = self._read_metadata()
        return int(metadata.get("total_cases", 0))

    def _load_system_error_count(self) -> int:
        metadata = self._read_metadata()
        return int(metadata.get("system_error_count", 0))

    def _find_incomplete_run(self) -> Path | None:
        candidates = sorted(
            [
                p
                for p in self.base_dir.iterdir()
                if p.is_dir() and p.name.startswith(self.config.name)
            ],
            reverse=True,
        )
        for run_path in candidates:
            metadata_path = run_path / "metadata.json"
            if not metadata_path.exists():
                continue
            data = {}
            with contextlib.suppress(json.JSONDecodeError), open(metadata_path) as f:
                data = json.load(f)
            if data.get("status") == "completed":
                continue
            if data.get("config_fingerprint") != self._config_fingerprint:
                continue
            return run_path
        return None

    def initialize(self):
        if not self.metadata_path.exists():
            self.run_dir.mkdir(parents=True, exist_ok=True)
            # Save config
            config_dict = self.base_config.model_dump()
            import yaml

            with open(self.config_path, "w") as f:
                yaml.dump(config_dict, f)

            # Init metadata
            self.update_metadata(
                {
                    "start_time": datetime.now().isoformat(),
                    "status": "running",
                    "config_name": self.config.name,
                    "config_fingerprint": self._config_fingerprint,
                    "completed_cases": 0,
                    "system_error_count": 0,
                }
            )
        else:
            # Resume/Retry
            self.update_metadata(
                {
                    "last_resumed": datetime.now().isoformat(),
                    "status": "running",
                    "completed_cases": self.completed_cases,
                    "system_error_count": self.system_error_count,
                }
            )

    def get_next_attempt(self, agent: str, dataset: str, case_id: str) -> int:
        """Get the next attempt number for a case."""
        return self._case_attempts.get((agent, dataset, case_id), 0) + 1

    def get_attempt_count(self, agent: str, dataset: str, case_id: str) -> int:
        """Get the current attempt count for a case."""
        return self._case_attempts.get((agent, dataset, case_id), 0)

    def save_result(self, result: CaseResult):
        entry = (result.agent_name, result.dataset_name, result.case_id)

        if result.timestamp is None:
            result.timestamp = datetime.now().isoformat()

        with open(self.results_path, "a") as f:
            f.write(result.model_dump_json() + "\n")

        if entry not in self._completed_entries:
            self._completed_entries.add(entry)
            self.completed_cases = len(self._completed_entries)
            self._write_progress()

        # Update attempts
        self._case_attempts[entry] = result.attempt

    def save_error(self, error_data: dict[str, Any], error_type: ErrorType = ErrorType.SYSTEM):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type.value,
            **error_data,
        }
        with open(self.errors_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
        if error_type == ErrorType.SYSTEM:
            self.system_error_count += 1
            self.update_metadata({"system_error_count": self.system_error_count})

    def save_dashboard_state(self, state_json: str):
        path = self.run_dir / "status.json"
        with open(path, "w") as f:
            f.write(state_json)

    def update_metadata(self, data: dict[str, Any]):
        # Read existing
        current = {}
        if self.metadata_path.exists():
            with open(self.metadata_path) as f, contextlib.suppress(json.JSONDecodeError):
                current = json.load(f)

        current.update(data)

        with open(self.metadata_path, "w") as f:
            json.dump(current, f, indent=2)

    def set_total_cases(self, total_cases: int):
        self.total_cases = total_cases
        self._write_progress()

    def mark_completed(self, failed: bool = False):
        status = "failed" if failed else "completed"
        self.update_metadata(
            {
                "status": status,
                "completed_time": datetime.now().isoformat(),
                "completed_cases": self.completed_cases,
            }
        )

    def load_existing_results(self) -> set[tuple[str, str, str]]:
        return set(self._completed_entries)

    def _load_completed_entries(self):
        if not self.results_path.exists():
            return
        with open(self.results_path) as f:
            for line in f:
                with contextlib.suppress(json.JSONDecodeError):
                    data = json.loads(line)
                    agent = data.get("agent_name", "")
                    dataset = data.get("dataset_name", "")
                    cid = data.get("case_id", "")
                    if agent and dataset and cid:
                        key = (agent, dataset, cid)
                        self._completed_entries.add(key)
                        # Track attempts
                        attempt = data.get("attempt")
                        if attempt:
                            self._case_attempts[key] = max(self._case_attempts[key], int(attempt))
                        else:
                            # Fallback: count occurrences
                            self._case_attempts[key] += 1

    def _write_progress(self):
        progress = None
        if self.total_cases:
            progress = self.completed_cases / self.total_cases
        payload = {"total_cases": self.total_cases, "completed_cases": self.completed_cases}
        if progress is not None:
            payload["progress"] = progress
        self.update_metadata(payload)

    def _read_metadata(self) -> dict[str, Any]:
        return self._read_metadata_file(self.metadata_path)

    @staticmethod
    def _read_metadata_file(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path) as f, contextlib.suppress(json.JSONDecodeError):
            return json.load(f)
        return {}


def _compute_config_fingerprint(config: BenchmarkConfig) -> str:
    normalized = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(normalized.encode()).hexdigest()


def find_latest_run(runs_dir: Path) -> Path | None:
    """Find the latest completed or failed run directory."""
    if not runs_dir.exists():
        return None

    candidates = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for run_path in candidates:
        metadata_path = run_path / "metadata.json"
        if not metadata_path.exists():
            continue

        data = RunManager._read_metadata_file(metadata_path)
        status = data.get("status")
        if status in ("completed", "failed"):
            return run_path

    return None
