import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pacabench.config import BenchmarkConfig, load_config

_RUNS_ENV_VAR = "PACABENCH_RUNS_DIR"


def _default_runs_dir() -> Path:
    env_dir = os.getenv(_RUNS_ENV_VAR)
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.cwd() / "runs"


def _default_dataset_cache_dir() -> Path:
    return Path.home() / ".cache" / "pacabench" / "datasets"


def resolve_runs_dir(config: BenchmarkConfig | None, override: Path | None = None) -> Path:
    if override:
        return override.expanduser()
    if config:
        return Path(config.output.directory).expanduser()
    return _default_runs_dir()


def resolve_runs_dir_from_cli(config_path: Path | None, override: Path | None = None) -> Path:
    cfg: BenchmarkConfig | None = None
    if config_path and config_path.exists():
        with contextlib.suppress(Exception):
            cfg = load_config(config_path)
    runs_dir = resolve_runs_dir(cfg, override)
    if config_path and not runs_dir.is_absolute():
        runs_dir = (config_path.parent / runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def resolve_run_directory(run_id: str, runs_dir: Path) -> Path:
    candidate = Path(run_id).expanduser()
    if candidate.exists():
        return candidate
    resolved = runs_dir / run_id
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Run directory not found for id '{run_id}' in {runs_dir}")


@dataclass
class EvalContext:
    config_path: Path
    base_config: BenchmarkConfig
    runtime_config: BenchmarkConfig
    runs_dir: Path
    datasets_cache_dir: Path
    root_dir: Path
    env: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> BenchmarkConfig:
        return self.runtime_config


def build_eval_context(
    config_path: Path,
    base_config: BenchmarkConfig,
    runtime_config: BenchmarkConfig,
    runs_dir_override: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> EvalContext:
    runs_dir = resolve_runs_dir(runtime_config, runs_dir_override)
    if not runs_dir.is_absolute():
        runs_dir = (config_path.parent / runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    datasets_cache_dir = _default_dataset_cache_dir()
    datasets_cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return EvalContext(
        config_path=config_path,
        base_config=base_config,
        runtime_config=runtime_config,
        runs_dir=runs_dir,
        datasets_cache_dir=datasets_cache_dir,
        root_dir=config_path.parent.resolve(),
        env=env,
        overrides=overrides or {},
    )
