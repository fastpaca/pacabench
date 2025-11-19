from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProxyConfig(BaseModel):
    enabled: bool = True
    provider: str = "openai"


class GlobalConfig(BaseModel):
    concurrency: int = 4
    timeout_seconds: float = 60.0
    max_retries: int = 2
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)


class AgentConfig(BaseModel):
    name: str
    command: str
    setup: str | None = None
    teardown: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class EvaluatorConfig(BaseModel):
    type: str
    model: str | None = None
    # Allow extra fields for flexibility
    extra_config: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class DatasetConfig(BaseModel):
    name: str
    source: str
    split: str | None = None
    prepare: str | None = None
    input_map: dict[str, str] = Field(default_factory=dict)
    evaluator: EvaluatorConfig | None = None


class OutputConfig(BaseModel):
    directory: str = "./runs"


class BenchmarkConfig(BaseModel):
    name: str
    description: str | None = None
    version: str = "0.1.0"
    author: str | None = None
    config: GlobalConfig = Field(default_factory=GlobalConfig)
    agents: list[AgentConfig] = Field(default_factory=list)
    datasets: list[DatasetConfig] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return BenchmarkConfig(**data)
