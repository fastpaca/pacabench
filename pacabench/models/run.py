"""Run-related models for PacaBench."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Status of a benchmark run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentMetadata(BaseModel):
    """Metadata about an agent in a run."""

    name: str
    command: str
    env_keys: list[str] = Field(default_factory=list)
    model: str | None = None


class DatasetMetadata(BaseModel):
    """Metadata about a dataset in a run."""

    name: str
    source: str
    split: str | None = None


class RunMetadata(BaseModel):
    """Persisted run state (metadata.json)."""

    run_id: str
    start_time: str
    status: RunStatus = RunStatus.RUNNING

    # Config info
    config_name: str
    config_description: str | None = None
    config_version: str | None = None
    config_path: str | None = None
    config_fingerprint: str | None = None

    # Progress
    total_cases: int = 0
    completed_cases: int = 0
    system_error_count: int = 0
    progress: float | None = None

    # Run details
    agents: list[AgentMetadata] = Field(default_factory=list)
    datasets: list[DatasetMetadata] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    completed_time: str | None = None
    last_resumed: str | None = None


class RunSummary(BaseModel):
    """Lightweight view of a run for listing."""

    run_id: str
    path: Path
    status: str  # Keep as str for backwards compat with existing runs
    start_time: str | None
    elapsed_seconds: float | None
    completed_cases: int
    total_cases: int
    progress: float | None
    datasets: list[str]
    agents: list[str]
    models: list[str]
    total_cost_usd: float | None

    class Config:
        arbitrary_types_allowed = True
