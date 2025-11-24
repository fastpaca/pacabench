import asyncio
import json
import logging
import os
import time

from pacabench.runners.base import BaseRunner
from pacabench.types import Case, ErrorType, RunnerMetrics, RunnerOutput

logger = logging.getLogger(__name__)


class CommandRunner(BaseRunner):
    def __init__(
        self, config, proxy_url: str | None = None, base_env: dict[str, str] | None = None
    ):
        super().__init__(config)
        self.proxy_url = proxy_url
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._base_env = base_env.copy() if base_env is not None else os.environ.copy()
        self._stderr_task: asyncio.Task | None = None

    async def start(self):
        if self._process:
            if self._process.returncode is None:
                return
            # Process is dead, clear it to allow restart
            await self.stop()

        env = self._base_env.copy()
        # Inject Agent env vars
        if self.config.env:
            env.update(self.config.env)

        # Inject Proxy URL
        if self.proxy_url:
            env["OPENAI_BASE_URL"] = self.proxy_url

        # Start process
        logger.info(f"Starting runner for agent {self.config.name}: {self.config.command}")

        # Setup command (run once) if provided
        if self.config.setup:
            logger.info(f"Running setup for {self.config.name}: {self.config.setup}")
            # Setup is usually blocking and separate
            proc = await asyncio.create_subprocess_shell(
                self.config.setup, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Setup failed for {self.config.name}")

        # Ensure stdout/stderr are buffered properly? default is typically fine.
        self._process = await asyncio.create_subprocess_shell(
            self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=1024 * 1024 * 10,
        )

        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def stop(self):
        if self._process:
            if self._process.returncode is None:
                try:
                    self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=5.0)
                    except TimeoutError:
                        self._process.kill()
                except ProcessLookupError:
                    pass
            self._process = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

        if self.config.teardown:
            logger.info(f"Running teardown for {self.config.name}: {self.config.teardown}")
            proc = await asyncio.create_subprocess_shell(
                self.config.teardown, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()

    async def run_case(self, case: Case) -> RunnerOutput:
        if not self._process or self._process.returncode is not None:
            # Restart if crashed
            logger.warning(f"Process for {self.config.name} is dead, restarting...")
            await self.start()

        async with self._lock:
            start_time = time.perf_counter()

            input_data = {
                **case.metadata,
                "case_id": case.case_id,
                "dataset_name": case.dataset_name,
                "agent_name": self.config.name,
                "input": case.input,
                "history": case.history,
            }

            json_line = json.dumps(input_data) + "\n"
            try:
                self._process.stdin.write(json_line.encode())
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                logger.error(f"Broken pipe for {self.config.name}")
                return RunnerOutput(
                    error="Process crashed (BrokenPipe)",
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                    error_type=ErrorType.SYSTEM,
                )

            # Read output
            while True:
                try:
                    line_bytes = await self._process.stdout.readline()
                    if not line_bytes:
                        # EOF
                        # Wait a brief moment for stderr to drain to logs
                        await asyncio.sleep(0.1)
                        return RunnerOutput(
                            error="Process exited unexpectedly (EOF)",
                            duration_ms=(time.perf_counter() - start_time) * 1000,
                            error_type=ErrorType.SYSTEM,
                        )

                    line = line_bytes.decode().strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        # Check if it's a result object
                        if "output" in data or "error" in data:
                            duration = (time.perf_counter() - start_time) * 1000
                            metrics = None
                            if "metrics" in data:
                                metrics = RunnerMetrics(**data["metrics"])

                            # Determine error type
                            error_msg = data.get("error")
                            error_type = ErrorType.NONE
                            if "error_type" in data:
                                try:
                                    error_type = ErrorType(data["error_type"])
                                except ValueError:
                                    logger.warning(
                                        f"Invalid error_type '{data['error_type']}' from agent {self.config.name}"
                                    )
                            elif error_msg:
                                # Default to SYSTEM if error is present but type is not specified
                                # This ensures generic errors are treated as retryable system failures
                                error_type = ErrorType.SYSTEM

                            return RunnerOutput(
                                output=data.get("output"),
                                error=error_msg,
                                metrics=metrics,
                                duration_ms=duration,
                                error_type=error_type,
                            )
                        else:
                            # Treat as log
                            logger.debug(f"[{self.config.name} LOG]: {line}")
                    except json.JSONDecodeError:
                        logger.debug(f"[{self.config.name} LOG]: {line}")

                except Exception as e:
                    duration = (time.perf_counter() - start_time) * 1000
                    logger.error(f"Error reading from runner {self.config.name}: {e}")
                    return RunnerOutput(
                        error=str(e), duration_ms=duration, error_type=ErrorType.SYSTEM
                    )

    async def _drain_stderr(self):
        if not self._process or not self._process.stderr:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.debug("[%s STDERR]: %s", self.config.name, line.decode().rstrip())
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Failed to drain stderr for %s: %s", self.config.name, exc)
