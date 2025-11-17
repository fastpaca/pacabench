"""Long-context agentic runner.

Baseline approach: Pass full context to LLM with tools.
"""

import time

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from smolagents.models import OpenAIServerModel

from agentbench.metrics import collect_metrics
from agentbench.runners.base import Runner, RunnerMetrics, RunnerOutput
from agentbench.types import Case, EvalContext


class LongContextAgenticRunner(Runner):
    """Long-context agentic runner with tools."""

    async def run_case(self, case: Case, ctx: EvalContext) -> RunnerOutput:
        """
        Run an agentic case with long context and tools.

        Args:
            case: Test case
            ctx: Evaluation context

        Returns:
            RunnerOutput with output, error, and metrics
        """
        start_time = time.time()

        try:
            model_obj = OpenAIServerModel(
                model_id=ctx.model,
                api_base=f"http://localhost:{ctx.proxy_port}/v1",
                api_key=ctx.openai_api_key,
            )
            tools = [
                DuckDuckGoSearchTool(),
                WikipediaSearchTool(),
                VisitWebpageTool(),
                PythonInterpreterTool(),
                FinalAnswerTool(),
            ]
            agent = CodeAgent(tools=tools, model=model_obj, max_steps=15, verbosity_level=0)

            question = case.inputs["question"]

            if case.inputs.get("file_name"):
                question += f"\n\nNote: A file '{case.inputs['file_name']}' is mentioned but not yet accessible."

            result = await agent.run(question, return_full_result=True)
            output = str(result.output).strip()

            duration_ms = (time.time() - start_time) * 1000
            llm_metrics = collect_metrics(ctx)
            metrics = RunnerMetrics(model_duration_ms=duration_ms, llm_metrics=llm_metrics)
            return RunnerOutput(output=output, error=None, metrics=metrics)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            llm_metrics = collect_metrics(ctx)
            metrics = RunnerMetrics(model_duration_ms=duration_ms, llm_metrics=llm_metrics)
            return RunnerOutput(
                output=None,
                error=f"Runner execution failed: {e}",
                metrics=metrics,
            )
