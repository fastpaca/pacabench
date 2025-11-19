import asyncio
import os
import time

from mem0 import Memory  # Sync memory
from pydantic_ai import Agent, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentbench.types import Case, Runner, RunnerContext, RunnerOutput

_AGENTS: dict[int, Agent] = {}


def _run_sync_memory_logic(case: Case, ctx: RunnerContext, proxy_url: str) -> str:
    """
    Synchronous wrapper for memory operations.
    Runs in a thread pool to avoid blocking the event loop.
    Initializes Memory fresh every time to avoid client leaks.
    """
    # Config
    qdrant_host = os.getenv("MEM0_QDRANT_HOST", "localhost")
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "agentbench",
                "host": qdrant_host,
                "port": 6333,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "openai_base_url": proxy_url,
                "api_key": ctx.openai_api_key,
                "http_client_proxies": None,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": ctx.model,
                "openai_base_url": proxy_url,
                "api_key": ctx.openai_api_key,
                "http_client_proxies": None,
            },
        },
    }

    # Init SYNC Memory
    m = Memory.from_config(config)

    user_id = case.id
    conversation = case.inputs["conversation"]
    question = case.inputs["question"]

    # Add
    if conversation:
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversation]
        m.add(messages, user_id=user_id)

    # Search
    search_result = m.search(
        query=question,
        user_id=user_id,
        limit=5,
    )

    relevant_memories = (
        search_result.get("results", [])
        if isinstance(search_result, dict)
        else (search_result if isinstance(search_result, list) else [])
    )

    memory_context = ""
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
            memory_context = "Relevant context from memory:\n" + "\n".join(memory_lines) + "\n\n"

    return memory_context


async def _get_agent(worker_id: int, ctx: RunnerContext) -> Agent:
    """Get or create Agent instance for a specific worker."""
    if worker_id not in _AGENTS:
        model_obj = OpenAIChatModel(
            ctx.model,
            provider=OpenAIProvider(
                base_url=f"http://localhost:{ctx.proxy_port}/v1",
                api_key=ctx.openai_api_key,
                http_client=None,  # Use default client
            ),
        )
        _AGENTS[worker_id] = Agent(model=model_obj)
    return _AGENTS[worker_id]


class Mem0Runner(Runner):
    """Mem0 QA runner with memory retrieval."""

    async def run_case(self, case: Case, ctx: RunnerContext) -> RunnerOutput:
        """
        Run a QA case with Mem0 memory retrieval.

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
                error="Runner execution failed: worker_id is required for Mem0Runner",
                duration_ms=(time.time() - start_time) * 1000,
            )

        try:
            # Offload SYNC memory operations to a thread
            proxy_url = f"http://localhost:{ctx.proxy_port}/v1"
            memory_context = await asyncio.get_running_loop().run_in_executor(
                None,  # Use default thread pool
                _run_sync_memory_logic,
                case,
                ctx,
                proxy_url,
            )

            agent = await _get_agent(ctx.worker_id, ctx)

            question = case.inputs["question"]
            choices = case.inputs.get("choices")

            message_history = [
                ModelRequest(parts=[UserPromptPart(content=memory_context)])
                if memory_context
                else None
            ]
            message_history = [m for m in message_history if m]

            prompt = f"Question: {question}"
            if choices:
                choices_text = "\n".join(
                    f"{key}. {value}" for key, value in sorted(choices.items())
                )
                prompt += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

            result = await agent.run(prompt, message_history=message_history)
            output = str(result.output)

            duration_ms = (time.time() - start_time) * 1000
            return RunnerOutput(output=output, error=None, duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return RunnerOutput(
                output=None,
                error=f"Runner execution failed: {e}",
                duration_ms=duration_ms,
            )
