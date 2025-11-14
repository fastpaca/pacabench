"""GAIA executor with mem0 memory layer for agentic workflows."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from mem0 import AsyncMemory
from pydantic_evals import increment_eval_metric
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from smolagents.models import OpenAIServerModel

from memharness.executors.base import Executor
from memharness.mem0 import patch_mem0_instrumentation

_MEMORY_RETRIEVAL_LIMIT = 5


def _extract_memories(search_result: Any) -> list[Any]:
    """Return search results as a list for both v0 and v1 mem0 responses."""
    if isinstance(search_result, dict):
        results = search_result.get("results")
        if isinstance(results, list):
            return results
        return []
    if isinstance(search_result, list):
        return search_result
    return []


class Mem0GAIAExecutor(Executor):
    """GAIA executor with mem0 memory layer.

    This executor augments the agentic workflow with long-term memory:
    1. Retrieves relevant memories based on the question
    2. Injects memory context into the agent's task
    3. Stores agent interactions for future retrieval
    """

    def __init__(
        self,
        model: str,
        provider: str,
        max_steps: int = 15,
        planning_interval: int | None = None,
        mem0_api_key: str | None = None,
        mem0_config: dict[str, Any] | None = None,
    ):
        """Initialize mem0 GAIA executor.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
            provider: Provider name (e.g., "openai", "anthropic")
            max_steps: Maximum number of agent steps (default: 15)
            planning_interval: Steps between replanning (None = no replanning)
            mem0_api_key: Optional API key for mem0
            mem0_config: Optional mem0 configuration dict
        """
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.planning_interval = planning_interval
        self.mem0_api_key = mem0_api_key or os.environ.get("MEM0_API_KEY")
        self.mem0_config = mem0_config
        self._agent = None
        self._memory: AsyncMemory | None = None

    def _build_memory_config(self) -> dict[str, Any]:
        if self.mem0_config is not None:
            config = deepcopy(self.mem0_config)
        else:
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "memharness_gaia",
                        "host": "memory",
                    },
                },
                "embedder": {"provider": "openai"},
            }
        if self.mem0_api_key:
            config["api_key"] = self.mem0_api_key
        return config

    async def _get_memory(self) -> AsyncMemory:
        """Lazily initialize shared memory client."""
        if self._memory is not None:
            return self._memory

        config = self._build_memory_config()
        self._memory = await AsyncMemory.from_config(config)
        return self._memory

    def _get_agent(self) -> CodeAgent:
        """Lazy initialize agent with tools (reused across all tasks)."""
        if self._agent is not None:
            return self._agent

        if self.provider == "openai":
            model_obj = OpenAIServerModel(
                model_id=self.model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif self.provider == "anthropic":
            model_obj = OpenAIServerModel(
                model_id=self.model,
                api_base="https://api.anthropic.com/v1",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        tools = [
            DuckDuckGoSearchTool(),
            WikipediaSearchTool(),
            VisitWebpageTool(),
            PythonInterpreterTool(),
            FinalAnswerTool(),
        ]

        self._agent = CodeAgent(
            tools=tools,
            model=model_obj,
            max_steps=self.max_steps,
            planning_interval=self.planning_interval,
            verbosity_level=0,
        )

        return self._agent

    async def execute(self, inputs: dict[str, Any]) -> str:
        """Execute GAIA task with mem0 memory augmentation.

        Args:
            inputs: Task inputs with 'question' key and optional 'case_id'

        Returns:
            String output (final answer)
        """
        agent = self._get_agent()
        memory = await self._get_memory()

        user_id = inputs.get("case_id", "default_user")
        question = inputs.get("question", "")

        search_result = await memory.search(
            query=question,
            user_id=user_id,
            limit=_MEMORY_RETRIEVAL_LIMIT,
        )
        relevant_memories = _extract_memories(search_result)

        increment_eval_metric("mem0_memories_retrieved", len(relevant_memories))

        augmented_question = question
        if relevant_memories:
            memory_lines = []
            for idx, mem in enumerate(relevant_memories, 1):
                if isinstance(mem, dict):
                    memory_text = mem.get("memory", "") or mem.get("text", "")
                elif isinstance(mem, str):
                    memory_text = mem
                else:
                    memory_text = str(mem)

                if memory_text:
                    memory_lines.append(f"{idx}. {memory_text}")

            if memory_lines:
                memory_context = (
                    "Relevant context from previous tasks:\n" + "\n".join(memory_lines) + "\n\n"
                )
                augmented_question = memory_context + question

        if inputs.get("file_name"):
            augmented_question += (
                f"\n\nNote: A file '{inputs['file_name']}' is mentioned but not yet accessible."
            )

        try:
            result = agent.run(augmented_question, return_full_result=True)

            increment_eval_metric("input_tokens", result.token_usage.input_tokens)
            increment_eval_metric("output_tokens", result.token_usage.output_tokens)
            increment_eval_metric("agent_steps", len(result.steps))
            increment_eval_metric("agent_runs", 1)

            final_answer = str(result.output).strip()

            await memory.add(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": final_answer},
                ],
                user_id=user_id,
            )
            increment_eval_metric("mem0_messages_stored", 2)

            return final_answer

        except Exception as e:
            increment_eval_metric("agent_errors", 1)
            return f"Error: {str(e)}"


patch_mem0_instrumentation()
