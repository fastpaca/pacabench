"""Mem0 agentic runner with tools and memory."""

import time
from pathlib import Path

from mem0 import AsyncMemory
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from smolagents.models import OpenAIServerModel

from agentbench.types import Case, Runner, RunnerContext, RunnerOutput

_MEMORIES: dict[int, AsyncMemory] = {}


async def _get_memory(worker_id: int, ctx: RunnerContext) -> AsyncMemory:
    """
    Get or create AsyncMemory instance for a specific worker.

    Lazy initialization: creates AsyncMemory on first use per worker.
    Each worker gets its own Chroma database path to avoid contention.
    Uses generic proxy URL since pipeline.py handles metrics aggregation via "_current".

    Args:
        worker_id: Stable worker identifier
        ctx: Runner context (for embedder config)

    Returns:
        AsyncMemory instance for this worker
    """
    if worker_id not in _MEMORIES:
        db_path = str(Path(f"/tmp/mem0-chroma-agentbench-gaia-worker-{worker_id}").resolve())
        proxy_url = f"http://localhost:{ctx.proxy_port}/v1"

        vector_store = {
            "provider": "chroma",
            "config": {
                "collection_name": "agentbench_gaia",
                "path": db_path,
            },
        }

        embedder_config = {
            "provider": "openai",
            "config": {
                "openai_base_url": proxy_url,
                "api_key": ctx.openai_api_key,
                "http_client_proxies": None,  # Disable proxies to prevent socket exhaustion
            },
        }

        llm_config = {
            "provider": "openai",
            "config": {
                "model": ctx.model,
                "openai_base_url": proxy_url,
                "api_key": ctx.openai_api_key,
                "http_client_proxies": None,  # Disable proxies to prevent socket exhaustion
            },
        }

        config = {
            "vector_store": vector_store,
            "embedder": embedder_config,
            "llm": llm_config,
        }
        _MEMORIES[worker_id] = await AsyncMemory.from_config(config)

    return _MEMORIES[worker_id]


class Mem0AgenticRunner(Runner):
    """Mem0 agentic runner with tools and memory."""

    async def run_case(self, case: Case, ctx: RunnerContext) -> RunnerOutput:
        """
        Run an agentic case with Mem0 memory and tools.

        Args:
            case: Test case
            ctx: Evaluation context (must have worker_id set)

        Returns:
            RunnerOutput with output, error, and metrics
        """
        start_time = time.time()

        if ctx.worker_id is None:
            return RunnerOutput(
                output=None,
                error="Runner execution failed: worker_id is required for Mem0AgenticRunner",
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            memory = await _get_memory(ctx.worker_id, ctx)
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

            user_id = case.id
            question = case.inputs["question"]

            search_result = await memory.search(query=question, user_id=user_id, limit=5)
            relevant_memories = (
                search_result.get("results", [])
                if isinstance(search_result, dict)
                else (search_result if isinstance(search_result, list) else [])
            )

            augmented_question = question
            if relevant_memories:
                memory_lines = []
                for idx, mem in enumerate(relevant_memories, 1):
                    memory_text = (
                        mem.get("memory", "") or mem.get("text", "")
                        if isinstance(mem, dict)
                        else (mem if isinstance(mem, str) else str(mem))
                    )
                    if memory_text:
                        memory_lines.append(f"{idx}. {memory_text}")
                if memory_lines:
                    memory_context = (
                        "Relevant context from previous tasks:\n" + "\n".join(memory_lines) + "\n\n"
                    )
                    augmented_question = memory_context + question

            if case.inputs.get("file_name"):
                augmented_question += f"\n\nNote: A file '{case.inputs['file_name']}' is mentioned but not yet accessible."

            result = await agent.run(augmented_question, return_full_result=True)
            answer = str(result.output).strip()
            await memory.add(
                [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                user_id=user_id,
            )

            duration_ms = (time.time() - start_time) * 1000
            return RunnerOutput(output=answer, error=None, duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return RunnerOutput(
                output=None,
                error=f"Runner execution failed: {e}",
                duration_ms=duration_ms,
            )
