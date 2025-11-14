"""Q&A executor with mem0 memory layer."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from httpx import AsyncClient, Limits
from mem0 import AsyncMemory
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import increment_eval_metric

from memharness.executors.base import Executor
from memharness.mem0 import patch_mem0_instrumentation

_MAX_CONNECTIONS = 20
_MAX_KEEPALIVE_CONNECTIONS = 50
_REQUEST_TIMEOUT_SECONDS = 300.0
_MEMORY_RETRIEVAL_LIMIT = 5

patch_mem0_instrumentation()


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


class Mem0QAExecutor(Executor):
    """Q&A executor with mem0 memory layer.

    This executor stores conversation history in mem0 and retrieves relevant
    memories to augment the LLM context. It follows mem0's benchmark approach
    of using keyword search + reranking for optimal retrieval performance.
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        mem0_api_key: str | None = None,
        mem0_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize mem0 Q&A executor.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
            provider: Provider name ("openai", "anthropic")
            base_url: Optional base URL for custom endpoints
            api_key: Optional API key for LLM provider
            mem0_api_key: Optional API key for mem0
            mem0_config: Optional mem0 configuration dict
            **kwargs: Additional configuration
        """
        super().__init__(model, provider, **kwargs)
        self.base_url = base_url
        self.api_key = api_key
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
                        "collection_name": "memharness",
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

    def _get_agent(self) -> Agent:
        """Lazily initialize and return agent."""
        if self._agent is not None:
            return self._agent

        http_client = AsyncClient(
            limits=Limits(
                max_connections=_MAX_CONNECTIONS,
                max_keepalive_connections=_MAX_KEEPALIVE_CONNECTIONS,
            ),
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

        if self.provider == "anthropic":
            model_obj = AnthropicModel(self.model)
        elif self.provider == "openai":
            model_obj = OpenAIChatModel(
                self.model,
                provider=OpenAIProvider(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    http_client=http_client,
                ),
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self._agent = Agent(model=model_obj)
        return self._agent

    async def execute(self, inputs: dict) -> str:
        """Execute Q&A task with mem0 memory augmentation.

        Args:
            inputs: Dict with:
                - conversation: List of message turns
                - question: Question to answer
                - choices: Optional multiple choice options
                - case_id: Optional case identifier for memory isolation

        Returns:
            Response string (choice letter for multiple choice, or answer text)
        """
        agent = self._get_agent()
        memory = await self._get_memory()

        user_id = inputs.get("case_id", "default_user")
        conversation = inputs.get("conversation", [])

        if conversation:
            messages = [{"role": turn["role"], "content": turn["content"]} for turn in conversation]
            await memory.add(messages, user_id=user_id)
            increment_eval_metric("mem0_messages_stored", len(messages))

        question = inputs["question"]
        search_result = await memory.search(
            query=question,
            user_id=user_id,
            limit=_MEMORY_RETRIEVAL_LIMIT,
        )
        relevant_memories = _extract_memories(search_result)

        increment_eval_metric("mem0_memories_retrieved", len(relevant_memories))

        memory_context = ""
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
                    "Relevant context from memory:\n" + "\n".join(memory_lines) + "\n\n"
                )

        prompt = memory_context + f"Question: {question}"

        if "choices" in inputs:
            choices_text = "\n".join(
                f"{key}. {value}" for key, value in sorted(inputs["choices"].items())
            )
            prompt += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

        result = await agent.run(prompt)

        usage = result.usage()
        increment_eval_metric("input_tokens", usage.input_tokens)
        increment_eval_metric("output_tokens", usage.output_tokens)
        increment_eval_metric("cache_write_tokens", usage.cache_write_tokens)
        increment_eval_metric("cache_read_tokens", usage.cache_read_tokens)

        return str(result.output)
