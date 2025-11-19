import asyncio
import logging
import os

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from agentbench.config import AgentConfig, BenchmarkConfig, DatasetConfig
from agentbench.datasets import get_dataset
from agentbench.evaluators import get_evaluator
from agentbench.persistence import RunManager
from agentbench.proxy import ProxyServer
from agentbench.runners.command import CommandRunner
from agentbench.types import Case, CaseResult, EvaluationResult, RunnerOutput

logger = logging.getLogger(__name__)


class Harness:
    def __init__(self, config: BenchmarkConfig, run_id: str | None = None):
        self.config = config
        self.run_manager = RunManager(config, run_id=run_id)
        self.proxies: list[ProxyServer] = []

    async def run(self, limit: int | None = None, whitelist_ids: set[str] | None = None):
        print(f"Starting benchmark: {self.config.name}")
        if whitelist_ids:
            print(f"Running only {len(whitelist_ids)} specific cases.")

        self.run_manager.initialize()

        # Load all datasets
        datasets = {}
        for ds_config in self.config.datasets:
            try:
                ds = get_dataset(ds_config)
                cases = ds.load()
                if limit:
                    cases = cases[:limit]
                datasets[ds_config.name] = (ds_config, cases)
                print(f"Loaded dataset {ds_config.name}: {len(cases)} cases")
            except Exception as e:
                logger.error(f"Failed to load dataset {ds_config.name}: {e}")

        # For each agent, for each dataset
        for agent_config in self.config.agents:
            for _, (ds_config, cases) in datasets.items():
                await self._run_dataset_agent(agent_config, ds_config, cases, whitelist_ids)

        # Cleanup global resources if any

    async def _run_dataset_agent(
        self,
        agent_config: AgentConfig,
        ds_config: DatasetConfig,
        cases: list[Case],
        whitelist_ids: set[str] | None = None,
    ):
        print(f"\nRunning Agent: {agent_config.name} on Dataset: {ds_config.name}")

        concurrency = self.config.config.concurrency

        completed_entries = self.run_manager.load_existing_results()

        # Filter cases
        pending_cases = []
        for c in cases:
            # If whitelist provided, only run if in whitelist
            if whitelist_ids is not None and c.case_id not in whitelist_ids:
                continue

            # If whitelist is None (normal run), skip completed
            # Check against (agent, dataset, case_id)
            entry = (agent_config.name, ds_config.name, c.case_id)
            if whitelist_ids is None and entry in completed_entries:
                continue

            pending_cases.append(c)

        if not pending_cases:
            print("All cases completed or skipped.")
            return

        # Setup Proxies
        proxies: list[ProxyServer] = []
        proxy_urls: list[str] = []

        if self.config.config.proxy.enabled:
            start_port = 8000
            for i in range(concurrency):
                port = start_port + i
                # Ideally use portpicker or handle port conflicts
                p = ProxyServer(port=port, openai_api_key=os.environ.get("OPENAI_API_KEY"))
                p.start()
                proxies.append(p)
                proxy_urls.append(f"http://127.0.0.1:{port}/v1")
        else:
            proxy_urls = [None] * concurrency

        # Setup Runners
        runners: list[CommandRunner] = []
        for i in range(concurrency):
            url = proxy_urls[i] if i < len(proxy_urls) else None
            runner = CommandRunner(agent_config, proxy_url=url)
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

        # Progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task_id = progress.add_task(
                f"Processing {len(pending_cases)} cases...", total=len(pending_cases)
            )

            async def worker(runner_idx: int):
                runner = runners[runner_idx]
                proxy = proxies[runner_idx] if runner_idx < len(proxies) else None

                while not queue.empty():
                    try:
                        case = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    try:
                        # Get Metrics from Proxy if enabled
                        llm_metrics = {}
                        if proxy:
                            proxy.metrics.clear_metrics("_current")

                        # Run (with timeout)
                        try:
                            runner_output = await asyncio.wait_for(
                                runner.run_case(case), timeout=self.config.config.timeout_seconds
                            )
                        except TimeoutError:
                            runner_output = RunnerOutput(
                                error="Timeout",
                                duration_ms=self.config.config.timeout_seconds * 1000,
                            )

                        if proxy:
                            proxy_metrics = proxy.metrics.get_metrics("_current")
                            llm_metrics = proxy_metrics

                        if runner_output.metrics:
                            # Merge logic
                            runner_metrics_dict = runner_output.metrics.model_dump(
                                exclude_none=True
                            )

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

                        # Evaluate
                        eval_result = EvaluationResult(passed=False, score=0.0)
                        if evaluator:
                            eval_result = await evaluator.evaluate(case, runner_output)

                        # Result
                        result = CaseResult(
                            case_id=case.case_id,
                            dataset_name=ds_config.name,
                            agent_name=agent_config.name,
                            passed=eval_result.passed,
                            output=runner_output.output,
                            error=runner_output.error,
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
                        )

                        self.run_manager.save_result(result)
                        progress.advance(task_id)

                    except Exception as e:
                        logger.error(f"System error processing case {case.case_id}: {e}")
                        self.run_manager.save_error({"case_id": case.case_id, "error": str(e)})
                        progress.advance(task_id)
                    finally:
                        queue.task_done()

            # Run workers
            await asyncio.gather(*[worker(i) for i in range(concurrency)])

        # Cleanup
        for runner in runners:
            await runner.stop()

        for proxy in proxies:
            proxy.stop()
