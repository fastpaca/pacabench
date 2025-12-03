"""Run management and discovery."""

import contextlib
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pacabench.models import (
    AgentMetadata,
    BenchmarkConfig,
    CaseResult,
    DatasetMetadata,
    ErrorType,
    RunMetadata,
    RunStatus,
    RunSummary,
)


class RunManager:
    """Manages run state and persistence."""

    def __init__(
        self,
        config: BenchmarkConfig,
        base_config: BenchmarkConfig,
        runs_dir: Path,
        config_path: Path,
        overrides: dict[str, Any] | None = None,
        run_id: str | None = None,
        force_new_run: bool = False,
    ):
        self.config = config
        self.base_config = base_config
        self.base_dir = runs_dir
        self.config_path = config_path
        self.overrides = overrides or {}
        self._config_fingerprint = _compute_config_fingerprint(config)
        self.resuming = False
        self._force_new_run = force_new_run

        if run_id:
            self.run_dir = self._resolve_run_dir(run_id)
            self.run_dir.parent.mkdir(parents=True, exist_ok=True)
            self.run_id = run_id
            metadata = self._load_metadata_file(self.run_dir / "metadata.json")
            if self.run_dir.exists() and metadata:
                fingerprint = metadata.config_fingerprint
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
            resume_dir = None if force_new_run else self.find_incomplete_run()
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
        self.saved_config_path = self.run_dir / "pacabench.yaml"
        self._completed_entries: set[tuple[str, str, str]] = set()
        self._case_attempts: dict[tuple[str, str, str], int] = defaultdict(int)
        self._load_completed_entries()
        self.completed_cases = len(self._completed_entries)
        self.total_cases = self._load_total_cases()
        self.system_error_count = self._load_system_error_count()

    def _generate_run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.config.name}-{ts}"

    def _resolve_run_dir(self, run_id: str) -> Path:
        candidate = Path(run_id)
        if candidate.is_absolute():
            return candidate
        return self.base_dir / run_id

    def _load_total_cases(self) -> int:
        metadata = self._load_metadata()
        return metadata.total_cases if metadata else 0

    def _load_system_error_count(self) -> int:
        metadata = self._load_metadata()
        return metadata.system_error_count if metadata else 0

    def find_incomplete_run(self) -> Path | None:
        """Find an incomplete run that can be resumed."""
        if not self.base_dir.exists():
            return None

        candidates = sorted(
            [
                p
                for p in self.base_dir.iterdir()
                if p.is_dir() and p.name.startswith(self.config.name)
            ],
            reverse=True,
        )
        for run_path in candidates:
            metadata = self._load_metadata_file(run_path / "metadata.json")
            if not metadata:
                continue
            if metadata.status == RunStatus.COMPLETED:
                continue
            if metadata.config_fingerprint != self._config_fingerprint:
                continue
            return run_path
        return None

    def initialize(self) -> None:
        """Initialize or resume a run."""
        agent_meta = [
            AgentMetadata(
                name=a.name,
                command=a.command,
                env_keys=sorted(a.env.keys()),
                model=a.env.get("OPENAI_MODEL"),
            )
            for a in self.config.agents
        ]
        dataset_meta = [
            DatasetMetadata(name=d.name, source=d.source, split=d.split)
            for d in self.config.datasets
        ]

        if not self.metadata_path.exists():
            self.run_dir.mkdir(parents=True, exist_ok=True)

            # Save config
            config_dict = self.base_config.model_dump()
            with open(self.saved_config_path, "w") as f:
                yaml.dump(config_dict, f)

            # Init metadata
            metadata = RunMetadata(
                run_id=self.run_id,
                start_time=datetime.now().isoformat(),
                status=RunStatus.RUNNING,
                config_name=self.config.name,
                config_description=self.config.description,
                config_version=self.config.version,
                config_path=str(self.config_path),
                config_fingerprint=self._config_fingerprint,
                agents=[m.model_dump() for m in agent_meta],
                datasets=[m.model_dump() for m in dataset_meta],
                overrides=self.overrides,
                completed_cases=0,
                system_error_count=0,
            )
            self._save_metadata(metadata)
        else:
            # Resume/Retry
            metadata = self._load_metadata()
            if metadata:
                metadata.last_resumed = datetime.now().isoformat()
                metadata.status = RunStatus.RUNNING
                metadata.completed_cases = self.completed_cases
                metadata.system_error_count = self.system_error_count
                metadata.agents = [m.model_dump() for m in agent_meta]
                metadata.datasets = [m.model_dump() for m in dataset_meta]
                self._save_metadata(metadata)

    def get_next_attempt(self, agent: str, dataset: str, case_id: str) -> int:
        """Get the next attempt number for a case."""
        return self._case_attempts.get((agent, dataset, case_id), 0) + 1

    def get_attempt_count(self, agent: str, dataset: str, case_id: str) -> int:
        """Get the current attempt count for a case."""
        return self._case_attempts.get((agent, dataset, case_id), 0)

    def save_result(self, result: CaseResult) -> None:
        """Save a case result to the results file."""
        entry = (result.agent_name, result.dataset_name, result.case_id)

        if result.timestamp is None:
            result.timestamp = datetime.now().isoformat()

        with open(self.results_path, "a") as f:
            f.write(result.model_dump_json() + "\n")

        if entry not in self._completed_entries:
            self._completed_entries.add(entry)
            self.completed_cases = len(self._completed_entries)
            self._write_progress()

        self._case_attempts[entry] = result.attempt

    def save_error(
        self, error_data: dict[str, Any], error_type: ErrorType = ErrorType.SYSTEM
    ) -> None:
        """Save an error to the errors file."""
        payload = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type.value,
            **error_data,
        }
        with open(self.errors_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
        if error_type == ErrorType.SYSTEM:
            self.system_error_count += 1
            self._update_metadata_field("system_error_count", self.system_error_count)

    def save_dashboard_state(self, state_json: str) -> None:
        """Save dashboard state for UI consumption."""
        path = self.run_dir / "status.json"
        with open(path, "w") as f:
            f.write(state_json)

    def set_total_cases(self, total_cases: int) -> None:
        """Set the total number of cases for progress tracking."""
        self.total_cases = total_cases
        self._write_progress()

    def mark_completed(self, failed: bool = False) -> None:
        """Mark the run as completed or failed."""
        status = RunStatus.FAILED if failed else RunStatus.COMPLETED
        metadata = self._load_metadata()
        if metadata:
            metadata.status = status
            metadata.completed_time = datetime.now().isoformat()
            metadata.completed_cases = self.completed_cases
            self._save_metadata(metadata)

    def load_existing_results(self) -> set[tuple[str, str, str]]:
        """Get the set of completed (agent, dataset, case_id) tuples."""
        return set(self._completed_entries)

    def _load_completed_entries(self) -> None:
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
                        attempt = data.get("attempt")
                        if attempt:
                            self._case_attempts[key] = max(self._case_attempts[key], int(attempt))
                        else:
                            self._case_attempts[key] += 1

    def _write_progress(self) -> None:
        metadata = self._load_metadata()
        if metadata:
            metadata.total_cases = self.total_cases
            metadata.completed_cases = self.completed_cases
            if self.total_cases:
                metadata.progress = self.completed_cases / self.total_cases
            self._save_metadata(metadata)

    def _load_metadata(self) -> RunMetadata | None:
        return self._load_metadata_file(self.metadata_path)

    def _save_metadata(self, metadata: RunMetadata) -> None:
        with open(self.metadata_path, "w") as f:
            f.write(metadata.model_dump_json(indent=2))

    def _update_metadata_field(self, field: str, value: Any) -> None:
        metadata = self._load_metadata()
        if metadata:
            setattr(metadata, field, value)
            self._save_metadata(metadata)

    @staticmethod
    def _load_metadata_file(path: Path) -> RunMetadata | None:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return RunMetadata.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return None


def _compute_config_fingerprint(config: BenchmarkConfig) -> str:
    """Compute a fingerprint for config comparison."""
    normalized = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(normalized.encode()).hexdigest()


def find_latest_run(runs_dir: Path) -> Path | None:
    """Find the latest run directory, including in-progress runs."""
    if not runs_dir.exists():
        return None

    candidates = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for run_path in candidates:
        metadata = RunManager._load_metadata_file(run_path / "metadata.json")
        if not metadata:
            continue
        if metadata.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.RUNNING):
            return run_path

    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def get_run_summaries(runs_dir: Path) -> list[RunSummary]:
    """Return run summaries sorted by start time, latest first."""
    if not runs_dir.exists():
        return []

    summaries: list[RunSummary] = []

    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue

        metadata = RunManager._load_metadata_file(run_path / "metadata.json")
        if not metadata:
            continue

        start_dt = _parse_datetime(metadata.start_time)
        end_str = metadata.completed_time or metadata.last_resumed
        end_dt = _parse_datetime(end_str) or (
            datetime.now() if metadata.status == RunStatus.RUNNING else None
        )
        elapsed = (end_dt - start_dt).total_seconds() if start_dt and end_dt else None

        progress = metadata.progress
        if progress is None and metadata.total_cases:
            progress = metadata.completed_cases / metadata.total_cases

        datasets = [d.name for d in metadata.datasets]
        agents = [a.name for a in metadata.agents]
        models = [a.model for a in metadata.agents if a.model]

        total_cost = None
        status_path = run_path / "status.json"
        if status_path.exists():
            with contextlib.suppress(Exception), open(status_path) as f:
                status_data = json.load(f)
                if isinstance(status_data, dict):
                    total_cost = status_data.get("total_cost")

        summaries.append(
            RunSummary(
                run_id=run_path.name,
                path=run_path,
                status=metadata.status,
                start_time=metadata.start_time,
                elapsed_seconds=elapsed,
                completed_cases=metadata.completed_cases,
                total_cases=metadata.total_cases,
                progress=progress,
                datasets=datasets,
                agents=agents,
                models=models,
                total_cost_usd=total_cost,
            )
        )

    summaries.sort(
        key=lambda s: (
            _parse_datetime(s.start_time) or datetime.fromtimestamp(s.path.stat().st_mtime)
        ),
        reverse=True,
    )
    return summaries


def get_failed_case_ids(run_dir: Path) -> set[str]:
    """Identify failed cases (system errors + wrong answers) in a run."""
    failed_ids: set[str] = set()

    # System errors
    errors_path = run_dir / "system_errors.jsonl"
    if errors_path.exists():
        with open(errors_path) as f:
            for line in f:
                try:
                    err = json.loads(line)
                    if "case_id" in err:
                        failed_ids.add(err["case_id"])
                except json.JSONDecodeError:
                    pass

    # Task failures
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    res = json.loads(line)
                    if not res.get("passed", False):
                        failed_ids.add(res["case_id"])
                except json.JSONDecodeError:
                    pass

    return failed_ids
