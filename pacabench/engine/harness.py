"""Main benchmark harness orchestrating runs."""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import portpicker

from pacabench.datasets import get_dataset
from pacabench.engine.dashboard import DashboardState
from pacabench.engine.proxy import ProxyServer
from pacabench.engine.reporters import ProgressReporter, RichProgressReporter
from pacabench.evaluators import get_evaluator
from pacabench.evaluators.base import BaseEvaluator
from pacabench.models import (
    AgentConfig,
    BenchmarkConfig,
    Case,
    CaseResult,
    DatasetConfig,
    ErrorType,
    EvaluationResult,
    RunnerOutput,
)
from pacabench.runners.command import CommandRunner
from pacabench.storage import RunManager

logger = logging.getLogger(__name__)


@dataclass
class CaseOutcome:
    """Result of processing a single case."""

    result: CaseResult
    cost: float
    passed: bool
    is_system_error: bool


class Harness:
    """Main benchmark harness that orchestrates running agents against datasets."""

    def __init__(
        self,
        config: BenchmarkConfig,
        base_config: BenchmarkConfig,
        runs_dir: Path,
        config_path: Path,
        env: dict[str, str],
        overrides: dict[str, Any] | None = None,
        run_id: str | None = None,
        force_new_run: bool = False,
        reporter: ProgressReporter | None = None,
    ):
        self.config = config
        self.base_config = base_config
        self.runs_dir = runs_dir
        self.config_path = config_path
        self.env = env
        self.overrides = overrides or {}

        self.run_manager = RunManager(
            config=config,
            base_config=base_config,
            runs_dir=runs_dir,
            config_path=config_path,
            overrides=self.overrides,
            run_id=run_id,
            force_new_run=force_new_run,
        )

        self._counts_lock = asyncio.Lock()
        self._processed_cases = 0
        self._system_error_cases = 0
        self._task_failures = 0
        self._circuit_open = False
        self.reporter = reporter or RichProgressReporter()

    async def run(
        self,
        limit: int | None = None,
        whitelist_ids: set[str] | None = None,
        datasets_cache_dir: Path | None = None,
    ) -> None:
        """Run the benchmark."""
        print(f"Starting benchmark: {self.config.name}")
        if whitelist_ids:
            print(f"Running only {len(whitelist_ids)} specific cases.")

        self.run_manager.initialize()
        print(f"Run directory: {self.run_manager.run_dir}")
        if self.run_manager.resuming:
            print(
                f"Resuming run {self.run_manager.run_id}: "
                f"{self.run_manager.completed_cases} cases previously completed."
            )

        datasets = self._load_datasets(limit, datasets_cache_dir)
        completed_entries = self.run_manager.load_existing_results()
        pending_case_map, total_pending, dashboard_state = self._prepare_run(
            datasets, whitelist_ids, completed_entries
        )

        self._set_total_cases(total_pending, whitelist_ids)

        await self.reporter.start(dashboard_state)

        try:
            for agent_config in self.config.agents:
                for _, (ds_config, _) in datasets.items():
                    pending_cases = pending_case_map.get((agent_config.name, ds_config.name), [])

                    dashboard_state.get_state(agent_config.name, ds_config.name).status = "Running"
                    await self.reporter.update(dashboard_state)

                    await self._run_dataset_agent(
                        agent_config, ds_config, pending_cases, dashboard_state
                    )

                    dashboard_state.get_state(
                        agent_config.name, ds_config.name
                    ).status = "Completed"
                    await self.reporter.update(dashboard_state)
                    self.run_manager.save_dashboard_state(dashboard_state.model_dump_json(indent=2))
        finally:
            await self.reporter.stop()

        self.run_manager.mark_completed(failed=self._circuit_open)

    def _load_datasets(
        self, limit: int | None, datasets_cache_dir: Path | None
    ) -> dict[str, tuple[DatasetConfig, list[Case]]]:
        """Load all datasets."""
        datasets: dict[str, tuple[DatasetConfig, list[Case]]] = {}
        cache_dir = datasets_cache_dir or Path.home() / ".cache" / "pacabench" / "datasets"

        for ds_config in self.config.datasets:
            try:
                ds = get_dataset(
                    ds_config, root_dir=self.config_path.parent, datasets_cache_dir=cache_dir
                )
                cases = ds.load(limit=limit)
                datasets[ds_config.name] = (ds_config, cases)
                print(f"Loaded dataset {ds_config.name}: {len(cases)} cases")
            except Exception as e:
                logger.error(f"Failed to load dataset {ds_config.name}: {e}")

        return datasets

    def _prepare_run(
        self,
        datasets: dict[str, tuple[DatasetConfig, list[Case]]],
        whitelist_ids: set[str] | None,
        completed_entries: set[tuple[str, str, str]],
    ) -> tuple[dict[tuple[str, str], list[Case]], int, DashboardState]:
        """Prepare pending cases and dashboard state."""
        pending_case_map: dict[tuple[str, str], list[Case]] = {}
        total_pending = 0
        dashboard_state = DashboardState()

        for agent_config in self.config.agents:
            for _, (ds_config, cases) in datasets.items():
                key = (agent_config.name, ds_config.name)
                pending = self._filter_pending_cases(
                    agent_config, ds_config, cases, whitelist_ids, completed_entries
                )
                pending_case_map[key] = pending
                total_pending += len(pending)

                dashboard_state.init_agent(
                    agent_config.name, ds_config.name, total_cases=len(cases)
                )
                state = dashboard_state.get_state(agent_config.name, ds_config.name)

                completed_count = sum(
                    1
                    for entry in completed_entries
                    if entry[0] == agent_config.name and entry[1] == ds_config.name
                )
                state.completed_cases = completed_count

        return pending_case_map, total_pending, dashboard_state

    def _set_total_cases(self, total_pending: int, whitelist_ids: set[str] | None) -> None:
        """Set total cases for progress tracking."""
        if self.run_manager.resuming and whitelist_ids is None:
            planned_total = self.run_manager.completed_cases + total_pending
        elif self.run_manager.resuming:
            planned_total = self.run_manager.total_cases or (
                self.run_manager.completed_cases + total_pending
            )
        else:
            planned_total = total_pending

        self.run_manager.set_total_cases(planned_total)

    async def _run_dataset_agent(
        self,
        agent_config: AgentConfig,
        ds_config: DatasetConfig,
        pending_cases: list[Case],
        state: DashboardState,
    ) -> None:
        """Run a single agent against a single dataset."""
        if not pending_cases:
            return

        concurrency = self.config.config.concurrency
        proxies, proxy_urls = self._setup_proxies(concurrency)
        runners = await self._setup_runners(agent_config, proxy_urls, concurrency)
        evaluator = get_evaluator(ds_config.evaluator) if ds_config.evaluator else None

        queue: asyncio.Queue[Case] = asyncio.Queue()
        for c in pending_cases:
            queue.put_nowait(c)

        recycle_interval = self.config.config.worker_recycle_interval
        tasks_since_restart = [0 for _ in range(concurrency)]

        async def worker(idx: int) -> None:
            await self._worker_loop(
                idx,
                queue,
                runners,
                proxies,
                proxy_urls,
                evaluator,
                agent_config,
                ds_config,
                state,
                tasks_since_restart,
                recycle_interval,
            )

        await asyncio.gather(*[worker(i) for i in range(concurrency)])

        for runner in runners:
            await runner.stop()
        for proxy in proxies:
            proxy.stop()

    def _setup_proxies(self, concurrency: int) -> tuple[list[ProxyServer], list[str | None]]:
        """Setup proxy servers for each worker."""
        proxies: list[ProxyServer] = []
        proxy_urls: list[str | None] = []

        if self.config.config.proxy.enabled:
            for _ in range(concurrency):
                port = portpicker.pick_unused_port()
                api_key = self.env.get("OPENAI_API_KEY")
                proxy_cfg = self.config.config.proxy
                p = ProxyServer(
                    port=port,
                    openai_api_key=api_key,
                    upstream_base_url=proxy_cfg.base_url,
                    provider=proxy_cfg.provider,
                )
                p.start()
                proxies.append(p)
                proxy_urls.append(f"http://127.0.0.1:{port}/v1")
        else:
            proxy_urls = [None] * concurrency

        return proxies, proxy_urls

    async def _setup_runners(
        self,
        agent_config: AgentConfig,
        proxy_urls: list[str | None],
        concurrency: int,
    ) -> list[CommandRunner]:
        """Setup command runners for each worker."""
        runners: list[CommandRunner] = []
        for i in range(concurrency):
            url = proxy_urls[i] if i < len(proxy_urls) else None
            runner = CommandRunner(agent_config, proxy_url=url, base_env=self.env)
            await runner.start()
            runners.append(runner)
        return runners

    async def _worker_loop(
        self,
        idx: int,
        queue: asyncio.Queue[Case],
        runners: list[CommandRunner],
        proxies: list[ProxyServer],
        proxy_urls: list[str | None],
        evaluator: BaseEvaluator | None,
        agent_config: AgentConfig,
        ds_config: DatasetConfig,
        state: DashboardState,
        tasks_since_restart: list[int],
        recycle_interval: int,
    ) -> None:
        """Worker loop that processes cases from the queue."""
        runner = runners[idx]
        proxy = proxies[idx] if idx < len(proxies) else None

        while not queue.empty():
            if self._circuit_open:
                break

            try:
                case = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                outcome = await self._process_case(
                    case, runner, proxy, proxy_urls[idx], evaluator, agent_config, ds_config
                )
                self._save_outcome(outcome, agent_config, ds_config)
                self._update_dashboard(outcome, state, agent_config.name, ds_config.name)
                await self._record_case_outcome(outcome.is_system_error, outcome.passed)
                await self.reporter.update(state)
                self.run_manager.save_dashboard_state(state.model_dump_json(indent=2))

                tasks_since_restart[idx] += 1
                if recycle_interval and tasks_since_restart[idx] >= recycle_interval:
                    await runners[idx].stop()
                    await runners[idx].start()
                    tasks_since_restart[idx] = 0

            except Exception as e:
                logger.error(f"System error processing case {case.case_id}: {e}")
                self.run_manager.save_error({"case_id": case.case_id, "error": str(e)})

                agent_state = state.get_state(agent_config.name, ds_config.name)
                agent_state.update_metrics(
                    passed=False, error=True, cost=0.0, latency_ms=0.0, case_id=case.case_id
                )
                await self.reporter.update(state)
                self.run_manager.save_dashboard_state(state.model_dump_json(indent=2))
                await self._record_case_outcome(is_system_error=True, passed=False)

            finally:
                queue.task_done()

            if self._circuit_open:
                state.circuit_open = True
                await self.reporter.update(state)
                self._drain_queue(queue)
                break

    async def _process_case(
        self,
        case: Case,
        runner: CommandRunner,
        proxy: ProxyServer | None,
        proxy_url: str | None,
        evaluator: BaseEvaluator | None,
        agent_config: AgentConfig,
        ds_config: DatasetConfig,
    ) -> CaseOutcome:
        """Process a single case and return the outcome."""
        llm_metrics: dict[str, Any] = {}

        if proxy:
            proxy.set_active_case(case.case_id)
            proxy.metrics.clear_metrics(case.case_id)

        attempt = self.run_manager.get_next_attempt(agent_config.name, ds_config.name, case.case_id)

        runner_output = await self._run_case_with_timeout(case, runner)

        if proxy:
            llm_metrics = proxy.metrics.get_metrics(case.case_id)

        if runner_output.metrics:
            self._merge_runner_metrics(runner_output, llm_metrics)

        eval_result = await self._evaluate_case(case, runner_output, evaluator, proxy_url)

        result = self._build_case_result(
            case, runner_output, eval_result, llm_metrics, agent_config, ds_config, attempt
        )

        is_system_error = runner_output.error_type == ErrorType.SYSTEM
        cost = 0.0
        passed = True

        if not is_system_error:
            cost = result.llm_metrics.get("llm_total_cost_usd", 0.0)
            if result.judge_cost_usd:
                cost += result.judge_cost_usd
            passed = result.passed

        return CaseOutcome(result=result, cost=cost, passed=passed, is_system_error=is_system_error)

    async def _run_case_with_timeout(self, case: Case, runner: CommandRunner) -> RunnerOutput:
        """Run a case with timeout handling."""
        start_time = time.time()
        try:
            runner_output = await asyncio.wait_for(
                runner.run_case(case), timeout=self.config.config.timeout_seconds
            )
        except TimeoutError:
            runner_output = RunnerOutput(
                error="Timeout",
                duration_ms=self.config.config.timeout_seconds * 1000,
                error_type=ErrorType.SYSTEM,
            )

        if runner_output.duration_ms == 0:
            runner_output.duration_ms = (time.time() - start_time) * 1000

        return runner_output

    def _merge_runner_metrics(
        self, runner_output: RunnerOutput, llm_metrics: dict[str, Any]
    ) -> None:
        """Merge runner-reported metrics into llm_metrics."""
        if not runner_output.metrics:
            return

        runner_metrics_dict = runner_output.metrics.model_dump(exclude_none=True)
        mapping = {
            "call_count": "llm_call_count",
            "input_tokens": "llm_input_tokens",
            "output_tokens": "llm_output_tokens",
            "cache_read_tokens": "llm_cache_read_tokens",
            "cache_write_tokens": "llm_cache_write_tokens",
            "cost_usd": "llm_total_cost_usd",
            "latency_ms": "llm_latency_ms",
        }

        for r_key, p_key in mapping.items():
            if r_key in runner_metrics_dict:
                val = runner_metrics_dict[r_key]
                llm_metrics[p_key] = [val] if r_key == "latency_ms" else val

    async def _evaluate_case(
        self,
        case: Case,
        runner_output: RunnerOutput,
        evaluator: BaseEvaluator | None,
        proxy_url: str | None,
    ) -> EvaluationResult:
        """Evaluate a case result."""
        if evaluator and runner_output.error_type != ErrorType.SYSTEM:
            return await evaluator.evaluate(case, runner_output, proxy_url=proxy_url)
        return EvaluationResult(passed=False, score=0.0)

    def _build_case_result(
        self,
        case: Case,
        runner_output: RunnerOutput,
        eval_result: EvaluationResult,
        llm_metrics: dict[str, Any],
        agent_config: AgentConfig,
        ds_config: DatasetConfig,
        attempt: int,
    ) -> CaseResult:
        """Build a CaseResult from components."""
        evaluator_type = ds_config.evaluator.type if ds_config.evaluator else None

        return CaseResult(
            case_id=case.case_id,
            dataset_name=ds_config.name,
            agent_name=agent_config.name,
            attempt=attempt,
            passed=eval_result.passed,
            output=runner_output.output,
            error=runner_output.error,
            error_type=runner_output.error_type,
            runner_duration_ms=runner_output.duration_ms,
            llm_metrics=llm_metrics,
            f1_score=eval_result.score if evaluator_type == "f1" else None,
            f1_passed=eval_result.passed if evaluator_type == "f1" else None,
            judge_passed=eval_result.passed if evaluator_type == "llm_judge" else None,
            judge_reason=eval_result.reason,
            judge_metrics=eval_result.metrics,
            judge_cost_usd=eval_result.metrics.get("cost_usd") if eval_result.metrics else None,
        )

    def _save_outcome(
        self, outcome: CaseOutcome, agent_config: AgentConfig, ds_config: DatasetConfig
    ) -> None:
        """Save the outcome to storage."""
        if outcome.is_system_error:
            self.run_manager.save_error(
                {
                    "case_id": outcome.result.case_id,
                    "agent_name": agent_config.name,
                    "dataset_name": ds_config.name,
                    "error": outcome.result.error,
                },
                error_type=ErrorType.SYSTEM,
            )
        else:
            self.run_manager.save_result(outcome.result)

    def _update_dashboard(
        self,
        outcome: CaseOutcome,
        state: DashboardState,
        agent_name: str,
        dataset_name: str,
    ) -> None:
        """Update dashboard state with outcome."""
        agent_state = state.get_state(agent_name, dataset_name)
        agent_state.update_metrics(
            passed=outcome.passed,
            error=outcome.is_system_error,
            cost=outcome.cost,
            latency_ms=outcome.result.runner_duration_ms,
            case_id=outcome.result.case_id,
        )
        state.total_cost += outcome.cost

    def _filter_pending_cases(
        self,
        agent_config: AgentConfig,
        ds_config: DatasetConfig,
        cases: list[Case],
        whitelist_ids: set[str] | None,
        completed_entries: set[tuple[str, str, str]],
    ) -> list[Case]:
        """Filter cases to only those pending execution."""
        pending = []
        for case in cases:
            if whitelist_ids is not None and case.case_id not in whitelist_ids:
                continue
            entry = (agent_config.name, ds_config.name, case.case_id)
            if whitelist_ids is None and entry in completed_entries:
                continue

            if whitelist_ids is None:
                current_attempts = self.run_manager.get_attempt_count(
                    agent_config.name, ds_config.name, case.case_id
                )
                max_allowed = self.config.config.max_retries + 1
                if current_attempts >= max_allowed:
                    continue

            pending.append(case)
        return pending

    async def _record_case_outcome(self, is_system_error: bool, passed: bool) -> None:
        """Record case outcome for circuit breaker."""
        async with self._counts_lock:
            self._processed_cases += 1
            if is_system_error:
                self._system_error_cases += 1
            elif not passed:
                self._task_failures += 1

            threshold = self.config.config.circuit_breaker_error_ratio or 0.0
            min_cases = self.config.config.circuit_breaker_min_cases

            if threshold > 0 and self._processed_cases >= min_cases and not self._circuit_open:
                ratio = self._system_error_cases / self._processed_cases
                if ratio >= threshold:
                    self._circuit_open = True
                    logger.error(
                        "Circuit breaker tripped: %.1f%% system errors (threshold %.1f%%).",
                        ratio * 100,
                        threshold * 100,
                    )

    def _drain_queue(self, queue: asyncio.Queue) -> None:
        """Drain remaining items from queue."""
        while not queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
                queue.task_done()
