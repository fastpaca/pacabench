import asyncio
import json
import logging
import os
import time

from agentbench.runners.base import BaseRunner
from agentbench.types import Case, RunnerMetrics, RunnerOutput

logger = logging.getLogger(__name__)


class CommandRunner(BaseRunner):
    def __init__(self, config, proxy_url: str | None = None):
        super().__init__(config)
        self.proxy_url = proxy_url
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def start(self):
        if self._process:
            return

        env = os.environ.copy()
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

        self._process = await asyncio.create_subprocess_shell(
            self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,  # We might want to capture stderr too
            env=env,
            limit=1024 * 1024 * 10,  # 10MB buffer for large JSONs
        )

        # Start a background task to log stderr?
        # For now, let's just focus on stdin/stdout loop.

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

            # Prepare input
            input_data = {
                "case_id": case.case_id,
                "dataset_name": case.dataset_name,
                "agent_name": self.config.name,
                "input": case.input,
                "history": case.history,
                **case.metadata,
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
                )

            # Read output
            while True:
                try:
                    # We need to read line by line.
                    # If process prints logs, we ignore them unless it's valid JSON with specific fields.
                    # Implement timeout? Harness level timeout vs Runner level.
                    # Spec says "timeout_seconds" in GlobalConfig.
                    # We should probably use `asyncio.wait_for`.

                    line_bytes = await self._process.stdout.readline()
                    if not line_bytes:
                        # EOF
                        return RunnerOutput(
                            error="Process exited unexpectedly (EOF)",
                            duration_ms=(time.perf_counter() - start_time) * 1000,
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

                            return RunnerOutput(
                                output=data.get("output"),
                                error=data.get("error"),
                                metrics=metrics,
                                duration_ms=duration,
                            )
                        else:
                            # Treat as log
                            logger.debug(f"[{self.config.name} LOG]: {line}")
                    except json.JSONDecodeError:
                        logger.debug(f"[{self.config.name} LOG]: {line}")

                except Exception as e:
                    duration = (time.perf_counter() - start_time) * 1000
                    logger.error(f"Error reading from runner {self.config.name}: {e}")
                    return RunnerOutput(error=str(e), duration_ms=duration)
