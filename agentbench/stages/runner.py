"""Stage 2: Execution - Subprocess runner for executing test cases."""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agentbench.stages.case import Case


@dataclass
class RunnerOutput:
    """Result from a runner execution."""

    result: str | None
    error: str | None
    duration_ms: float


class RunnerError(Exception):
    """Exception raised when runner execution fails."""

    pass


def spawn_runner(
    case: Case,
    runner_script: str,
    env: dict[str, str],
    timeout: float = 300.0,
) -> RunnerOutput:
    """
    Spawn a runner subprocess and execute a test case.

    Args:
        case: Test case to execute
        runner_script: Path to runner script (e.g., "runners/qa/long_context_runner.py")
        env: Environment variables to pass to runner
        timeout: Maximum execution time in seconds

    Returns:
        RunnerOutput with output or error

    Raises:
        RunnerError: If runner execution fails
    """
    if not runner_script.endswith("_runner.py"):
        runner_script = runner_script.replace(".py", "_runner.py")

    runner_path = Path(__file__).parent.parent.parent / runner_script

    if not runner_path.exists():
        raise RunnerError(f"Runner script not found: {runner_path}")

    case_data = {
        "id": case.id,
        "task_type": case.task_type,
        "inputs": case.inputs,
    }

    start_time = time.time()

    try:
        process = subprocess.Popen(
            [sys.executable, str(runner_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        stdout, stderr = process.communicate(
            input=json.dumps(case_data),
            timeout=timeout,
        )

        duration_ms = (time.time() - start_time) * 1000

        if process.returncode != 0:
            return RunnerOutput(
                result=None,
                error=f"Runner exited with code {process.returncode}: {stderr}",
                duration_ms=duration_ms,
            )

        try:
            output = json.loads(stdout)
            return RunnerOutput(
                result=output.get("result"),
                error=output.get("error"),
                duration_ms=duration_ms,
            )
        except json.JSONDecodeError:
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        output = json.loads(line)
                        return RunnerOutput(
                            result=output.get("result"),
                            error=output.get("error"),
                            duration_ms=duration_ms,
                        )
                    except json.JSONDecodeError:
                        continue

            return RunnerOutput(
                result=None,
                error=f"Failed to find valid JSON in runner output.\nStdout: {stdout[:500]}\nStderr: {stderr}",
                duration_ms=duration_ms,
            )

    except subprocess.TimeoutExpired:
        process.kill()
        duration_ms = (time.time() - start_time) * 1000
        return RunnerOutput(
            result=None,
            error=f"Runner timeout after {timeout}s",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return RunnerOutput(
            result=None,
            error=f"Runner execution failed: {e}",
            duration_ms=duration_ms,
        )
