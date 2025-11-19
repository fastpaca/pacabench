import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agentbench.config import BenchmarkConfig
from agentbench.types import CaseResult


class RunManager:
    def __init__(self, config: BenchmarkConfig, run_id: str | None = None):
        self.config = config
        self.base_dir = Path(config.output.directory)

        if run_id:
            self.run_id = run_id
            self.run_dir = self.base_dir / self.run_id
            if not self.run_dir.exists():
                raise ValueError(f"Run directory {self.run_dir} does not exist.")
        else:
            self.run_id = self._generate_run_id()
            self.run_dir = self.base_dir / self.run_id

        self.results_path = self.run_dir / "results.jsonl"
        self.errors_path = self.run_dir / "system_errors.jsonl"
        self.metadata_path = self.run_dir / "metadata.json"
        self.config_path = self.run_dir / "agentbench.yaml"

    def _generate_run_id(self) -> str:
        # format: dataset-agent-timestamp
        # But config has multiple datasets/agents.
        # Use config name + timestamp
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.config.name}-{ts}"

    def initialize(self):
        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True, exist_ok=True)
            # Save config
            config_dict = self.config.model_dump()
            import yaml

            with open(self.config_path, "w") as f:
                yaml.dump(config_dict, f)

            # Init metadata
            self.update_metadata(
                {
                    "start_time": datetime.now().isoformat(),
                    "status": "running",
                    "config_name": self.config.name,
                }
            )
        else:
            # Resume/Retry
            self.update_metadata({"last_resumed": datetime.now().isoformat(), "status": "running"})

    def save_result(self, result: CaseResult):
        with open(self.results_path, "a") as f:
            f.write(result.model_dump_json() + "\n")

    def save_error(self, error_data: dict[str, Any]):
        with open(self.errors_path, "a") as f:
            f.write(json.dumps(error_data) + "\n")

    def update_metadata(self, data: dict[str, Any]):
        # Read existing
        current = {}
        if self.metadata_path.exists():
            with open(self.metadata_path) as f, contextlib.suppress(json.JSONDecodeError):
                current = json.load(f)

        current.update(data)

        with open(self.metadata_path, "w") as f:
            json.dump(current, f, indent=2)

    def load_existing_results(self) -> set[tuple[str, str, str]]:
        # Return set of (agent_name, dataset_name, case_id) tuples
        completed = set()
        if self.results_path.exists():
            with open(self.results_path) as f:
                for line in f:
                    with contextlib.suppress(json.JSONDecodeError):
                        data = json.loads(line)
                        # Fallback for older logs if needed, but we just started
                        agent = data.get("agent_name", "")
                        dataset = data.get("dataset_name", "")
                        cid = data.get("case_id", "")
                        if agent and dataset and cid:
                            completed.add((agent, dataset, cid))
        return completed
