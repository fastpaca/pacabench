import asyncio
import contextlib
import logging
import time

import portpicker
from rich.live import Live

from agentbench.config import AgentConfig, DatasetConfig
from agentbench.context import EvalContext
from agentbench.dashboard import DashboardRenderer, DashboardState
from agentbench.datasets import get_dataset
from agentbench.evaluators import get_evaluator
from agentbench.persistence import RunManager
from agentbench.proxy import ProxyServer
from agentbench.runners.command import CommandRunner
from agentbench.types import Case, CaseResult, ErrorType, EvaluationResult, RunnerOutput

logger = logging.getLogger(__name__)


class Harness:
    def __init__(self, ctx: EvalContext, run_id: str | None = None, force_new_run: bool = False):
        self.ctx = ctx
        self.config = ctx.runtime_config
        self.run_manager = RunManager(ctx, run_id=run_id, force_new_run=force_new_run)
        self.proxies: list[ProxyServer] = []
        self._counts_lock = asyncio.Lock()
        self._processed_cases = 0
        self._system_error_cases = 0
        self._task_failures = 0
        self._total_cost_usd = 0.0
        self._circuit_open = False

    async def run(self, limit: int | None = None, whitelist_ids: set[str] | None = None):
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

        # Load all datasets
        datasets = {}
        for ds_config in self.config.datasets:
            try:
                ds = get_dataset(ds_config, self.ctx)
                cases = ds.load(limit=limit)
                datasets[ds_config.name] = (ds_config, cases)
                print(f"Loaded dataset {ds_config.name}: {len(cases)} cases")
            except Exception as e:
                logger.error(f"Failed to load dataset {ds_config.name}: {e}")

        completed_entries = self.run_manager.load_existing_results()
        pending_case_map: dict[tuple[str, str], list[Case]] = {}
        total_pending = 0

        # Initialize Dashboard State
        dashboard_state = DashboardState()
        dashboard_renderer = DashboardRenderer()

        for agent_config in self.config.agents:
            for _, (ds_config, cases) in datasets.items():
                key = (agent_config.name, ds_config.name)
                pending = self._filter_pending_cases(
                    agent_config, ds_config, cases, whitelist_ids, completed_entries
                )
                pending_case_map[key] = pending
                total_pending += len(pending)

                # Initialize agent state in dashboard
                # Use total cases from dataset for the progress bar total
                # (or just pending if we want to show work remaining)
                # Let's use total loaded cases to show overall progress
                dashboard_state.init_agent(
                    agent_config.name, ds_config.name, total_cases=len(cases)
                )
                # Mark already completed cases in dashboard state
                state = dashboard_state.get_state(agent_config.name, ds_config.name)

                # We need to count how many were completed for this agent/dataset
                # This is a bit expensive to scan completed_entries, but okay for init
                completed_count = 0
                for entry in completed_entries:
                    if entry[0] == agent_config.name and entry[1] == ds_config.name:
                        completed_count += 1
                state.completed_cases = completed_count
                # Ideally we would restore passed/failed/cost counts too if we want accurate resumption stats
                # But parsing results.jsonl again is slow. Let's accept 0 start for resumed stats or...
                # For now, leave cost/pass rate as "Session Stats" or accept they start at 0.
                # Correctness: The user wants "Snapshot of current state".
                # If we resume, we want to see TOTAL progress.
                # Let's just count completed for progress bar. Pass rate will be for "this run" unless we parse everything.
                # TODO: Load full stats from results.jsonl if possible.

        if self.run_manager.resuming and whitelist_ids is None:
            planned_total = self.run_manager.completed_cases + total_pending
        else:
            if self.run_manager.resuming:
                planned_total = self.run_manager.total_cases or (
                    self.run_manager.completed_cases + total_pending
                )
            else:
                planned_total = total_pending

        self.run_manager.set_total_cases(planned_total)

        # Live Dashboard
        with Live(dashboard_renderer.render(dashboard_state), refresh_per_second=4) as live:
            # For each agent, for each dataset
            for agent_config in self.config.agents:
                for _, (ds_config, _cases) in datasets.items():
                    pending_cases = pending_case_map.get((agent_config.name, ds_config.name), [])

                    # Update status to Running
                    dashboard_state.get_state(agent_config.name, ds_config.name).status = "Running"
                    live.update(dashboard_renderer.render(dashboard_state))

                    await self._run_dataset_agent(
                        agent_config,
                        ds_config,
                        pending_cases,
                        live,
                        dashboard_renderer,
                        dashboard_state,
                        whitelist_ids,
                    )

                    # Update status to Completed
                    dashboard_state.get_state(
                        agent_config.name, ds_config.name
                    ).status = "Completed"
                    live.update(dashboard_renderer.render(dashboard_state))
                    self.run_manager.save_dashboard_state(dashboard_state.model_dump_json(indent=2))

        self.run_manager.mark_completed(failed=self._circuit_open)

    async def _run_dataset_agent(
        self,
        agent_config: AgentConfig,
        ds_config: DatasetConfig,
        pending_cases: list[Case],
        live: Live,
        renderer: DashboardRenderer,
        state: DashboardState,
        whitelist_ids: set[str] | None = None,
    ):
        concurrency = self.config.config.concurrency

        if not pending_cases:
            return

        # Setup Proxies
        proxies: list[ProxyServer] = []
        proxy_urls: list[str] = []

        if self.config.config.proxy.enabled:
            for _ in range(concurrency):
                port = portpicker.pick_unused_port()
                api_key = self.ctx.env.get("OPENAI_API_KEY")
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

        # Setup Runners
        runners: list[CommandRunner] = []
        for i in range(concurrency):
            url = proxy_urls[i] if i < len(proxy_urls) else None
            runner = CommandRunner(agent_config, proxy_url=url, base_env=self.ctx.env)
            await runner.start()
            runners.append(runner)

        # Setup Evaluator
        evaluator = None
        if ds_config.evaluator:
            evaluator = get_evaluator(ds_config.evaluator)

        # Queue cases
        queue = asyncio.Queue()
        for c in pending_cases:
            queue.put_nowait(c)

        recycle_interval = self.config.config.worker_recycle_interval
        tasks_since_restart = [0 for _ in range(concurrency)]

        async def worker(runner_idx: int):
            runner = runners[runner_idx]
            proxy = proxies[runner_idx] if runner_idx < len(proxies) else None

            while not queue.empty():
                if self._circuit_open:
                    break
                try:
                    case = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                try:
                    # Get Metrics from Proxy if enabled
                    llm_metrics = {}
                    if proxy:
                        proxy.set_active_case(case.case_id)
                        proxy.metrics.clear_metrics(case.case_id)

                    # Get attempt number
                    attempt = self.run_manager.get_next_attempt(
                        agent_config.name, ds_config.name, case.case_id
                    )

                    # Run (with timeout)
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

                    # Determine latency if not provided
                    if runner_output.duration_ms == 0:
                        runner_output.duration_ms = (time.time() - start_time) * 1000

                    if proxy:
                        proxy_metrics = proxy.metrics.get_metrics(case.case_id)
                        llm_metrics = proxy_metrics

                    if runner_output.metrics:
                        # Merge logic
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
                                if r_key == "latency_ms":
                                    llm_metrics[p_key] = [val]
                                else:
                                    llm_metrics[p_key] = val

                    eval_result = EvaluationResult(passed=False, score=0.0)
                    if evaluator and runner_output.error_type != ErrorType.SYSTEM:
                        eval_result = await evaluator.evaluate(
                            case, runner_output, proxy_url=proxy_urls[runner_idx]
                        )

                    # Result
                    result = CaseResult(
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
                        f1_score=eval_result.score
                        if ds_config.evaluator and ds_config.evaluator.type == "f1"
                        else None,
                        f1_passed=eval_result.passed
                        if ds_config.evaluator and ds_config.evaluator.type == "f1"
                        else None,
                        judge_passed=eval_result.passed
                        if ds_config.evaluator and ds_config.evaluator.type == "llm_judge"
                        else None,
                        judge_reason=eval_result.reason,
                        judge_metrics=eval_result.metrics,
                        judge_cost_usd=eval_result.metrics.get("cost_usd")
                        if eval_result.metrics
                        else None,
                    )

                    if runner_output.error_type == ErrorType.SYSTEM:
                        self.run_manager.save_error(
                            {
                                "case_id": case.case_id,
                                "agent_name": agent_config.name,
                                "dataset_name": ds_config.name,
                                "error": runner_output.error,
                            },
                            error_type=runner_output.error_type,
                        )
                    else:
                        self.run_manager.save_result(result)

                    # Calculate cost and passed
                    cost = 0.0
                    passed = True
                    if runner_output.error_type != ErrorType.SYSTEM:
                        cost = result.llm_metrics.get("llm_total_cost_usd", 0.0)
                        if result.judge_cost_usd:
                            cost += result.judge_cost_usd
                        passed = result.passed

                    # Update Dashboard State
                    agent_state = state.get_state(agent_config.name, ds_config.name)
                    agent_state.update_metrics(
                        passed=passed,
                        error=(runner_output.error_type == ErrorType.SYSTEM),
                        cost=cost,
                        latency_ms=runner_output.duration_ms,
                        case_id=case.case_id,
                    )

                    state.total_cost += cost

                    await self._record_case_outcome(
                        runner_output.error_type,
                        passed=passed,
                        agent_name=agent_config.name,
                        dataset_name=ds_config.name,
                    )

                    # Refresh Live Dashboard
                    live.update(renderer.render(state))

                    # Save Snapshot
                    self.run_manager.save_dashboard_state(state.model_dump_json(indent=2))

                    tasks_since_restart[runner_idx] += 1
                    if (
                        recycle_interval
                        and recycle_interval > 0
                        and tasks_since_restart[runner_idx] >= recycle_interval
                    ):
                        await runners[runner_idx].stop()
                        await runners[runner_idx].start()
                        tasks_since_restart[runner_idx] = 0

                except Exception as e:
                    logger.error(f"System error processing case {case.case_id}: {e}")
                    self.run_manager.save_error({"case_id": case.case_id, "error": str(e)})

                    # Update Dashboard on System Error Exception
                    agent_state = state.get_state(agent_config.name, ds_config.name)
                    agent_state.update_metrics(
                        passed=False, error=True, cost=0.0, latency_ms=0.0, case_id=case.case_id
                    )
                    live.update(renderer.render(state))
                    self.run_manager.save_dashboard_state(state.model_dump_json(indent=2))

                    await self._record_case_outcome(
                        ErrorType.SYSTEM,
                        passed=False,
                        agent_name=agent_config.name,
                        dataset_name=ds_config.name,
                    )
                finally:
                    queue.task_done()
                if self._circuit_open:
                    state.circuit_open = True
                    live.update(renderer.render(state))
                    self._drain_queue(queue)
                    break

        # Run workers
        await asyncio.gather(*[worker(i) for i in range(concurrency)])

        # Cleanup
        for runner in runners:
            await runner.stop()

        for proxy in proxies:
            proxy.stop()

    def _filter_pending_cases(
        self,
        agent_config: AgentConfig,
        ds_config: DatasetConfig,
        cases: list[Case],
        whitelist_ids: set[str] | None,
        completed_entries: set[tuple[str, str, str]],
    ) -> list[Case]:
        pending = []
        for case in cases:
            if whitelist_ids is not None and case.case_id not in whitelist_ids:
                continue
            entry = (agent_config.name, ds_config.name, case.case_id)
            if whitelist_ids is None and entry in completed_entries:
                continue

            # Check max retries (only if not explicit retry)
            if whitelist_ids is None:
                current_attempts = self.run_manager.get_attempt_count(
                    agent_config.name, ds_config.name, case.case_id
                )
                # max_retries = N means we allow 1 initial + N retries = N+1 total attempts
                max_allowed = self.config.config.max_retries + 1
                if current_attempts >= max_allowed:
                    continue

            pending.append(case)
        return pending

    async def _record_case_outcome(
        self,
        error_type: ErrorType,
        passed: bool = True,
        agent_name: str | None = None,
        dataset_name: str | None = None,
    ):
        async with self._counts_lock:
            self._processed_cases += 1
            if error_type == ErrorType.SYSTEM:
                self._system_error_cases += 1
            elif not passed:
                self._task_failures += 1

            processed = self._processed_cases
            system_errors = self._system_error_cases
            threshold = self.config.config.circuit_breaker_error_ratio or 0.0
            min_cases = self.config.config.circuit_breaker_min_cases
            if threshold and threshold > 0 and processed >= min_cases and not self._circuit_open:
                ratio = system_errors / processed
                if ratio >= threshold:
                    self._circuit_open = True
                    logger.error(
                        "Circuit breaker tripped: %.1f%% system errors (threshold %.1f%%).",
                        ratio * 100,
                        threshold * 100,
                    )

    def _drain_queue(self, queue: asyncio.Queue):
        while not queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
                queue.task_done()
