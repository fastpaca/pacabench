"""Mem0 QA runner with memory retrieval."""

import time
from pathlib import Path

from mem0 import AsyncMemory
from pydantic_ai import Agent, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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
        db_path = str(Path(f"/tmp/mem0-chroma-agentbench-worker-{worker_id}").resolve())
        proxy_url = f"http://localhost:{ctx.proxy_port}/v1"

        vector_store = {
            "provider": "chroma",
            "config": {
                "collection_name": "agentbench",
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
        if ctx.embedding_model:
            embedder_config["config"]["model"] = ctx.embedding_model

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
            memory = await _get_memory(ctx.worker_id, ctx)
            model_obj = OpenAIChatModel(
                ctx.model,
                provider=OpenAIProvider(
                    base_url=f"http://localhost:{ctx.proxy_port}/v1",
                    api_key=ctx.openai_api_key,
                ),
            )
            agent = Agent(model=model_obj)

            user_id = case.id
            conversation = case.inputs["conversation"]
            question = case.inputs["question"]
            choices = case.inputs.get("choices")

            if conversation:
                messages = [
                    {"role": msg["role"], "content": msg["content"]} for msg in conversation
                ]
                await memory.add(messages, user_id=user_id)

            search_result = await memory.search(
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
                    memory_context = (
                        "Relevant context from memory:\n" + "\n".join(memory_lines) + "\n\n"
                    )

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
