"""Mem0 memory service answerer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel


@dataclass
class AnswerResult:
    """Result from an answerer."""

    response: str
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    metrics: dict[str, Any] = field(default_factory=dict)


class Mem0Answerer:
    """Answerer using Mem0 memory service.

    Mem0 manages conversation memory and retrieval, providing relevant
    context to augment the generation process.
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        mem0_api_key: str | None = None,
    ):
        """Initialize Mem0 answerer.

        Args:
            model: Model name
            provider: Provider name ("openai", "anthropic")
            base_url: Optional base URL for custom endpoints
            api_key: Optional API key for LLM
            mem0_api_key: Mem0 API key
        """
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.mem0_api_key = mem0_api_key
        self._agent = None  # Lazy initialization

        # TODO: Initialize Mem0 client
        # from mem0 import Memory
        # self.mem0_client = Memory(api_key=mem0_api_key)

    @property
    def agent(self) -> Agent:
        """Lazy-load the agent."""
        if self._agent is None:
            if self.provider == "anthropic":
                model = AnthropicModel(self.model, api_key=self.api_key)
            elif self.provider == "openai":
                model = OpenAIChatModel(
                    self.model,
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            self._agent = Agent(model=model)
        return self._agent

    async def answer(self, sample: Any) -> AnswerResult:
        """Generate an answer using Mem0 memory.

        Multi-phase process:
        1. Store conversation in Mem0
        2. Retrieve relevant memories
        3. Generate answer with augmented context

        Args:
            sample: Dataset sample with conversation, question, and choices

        Returns:
            AnswerResult with response and multi-phase metrics
        """
        total_start = time.time()

        # Phase 1: Store conversation
        store_start = time.time()
        user_id = self._get_user_id(sample)
        # TODO: Store conversation in Mem0
        # for turn in sample.conversation:
        #     self.mem0_client.add(
        #         messages=[turn],
        #         user_id=user_id
        #     )
        store_latency_ms = (time.time() - store_start) * 1000

        # Phase 2: Retrieve relevant memories
        retrieve_start = time.time()
        # TODO: Retrieve relevant context
        # context = self.mem0_client.search(
        #     query=sample.question,
        #     user_id=user_id,
        #     limit=10
        # )
        context = []  # Stub
        retrieve_latency_ms = (time.time() - retrieve_start) * 1000

        # Phase 3: Generate answer
        gen_start = time.time()
        prompt = self._format_prompt(sample, context)
        result = await self.agent.run(prompt)
        gen_latency_ms = (time.time() - gen_start) * 1000

        # Extract response
        if hasattr(result.output, "choice"):
            response = result.output.choice
        else:
            response = str(result.output)

        # Get token usage
        usage = result.usage()

        total_latency_ms = (time.time() - total_start) * 1000

        return AnswerResult(
            response=response,
            total_latency_ms=total_latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            metrics={
                "store_latency_ms": store_latency_ms,
                "retrieve_latency_ms": retrieve_latency_ms,
                "generation_latency_ms": gen_latency_ms,
                "num_retrieved_memories": len(context),
            },
        )

    def _get_user_id(self, sample: Any) -> str:
        """Generate a user ID for the sample."""
        return f"user_{sample.id}"

    def _format_prompt(self, sample: Any, context: list) -> str:
        """Format prompt with retrieved context."""
        parts = []

        # Add retrieved context
        if context:
            parts.append("Relevant memories:\n")
            for item in context:
                parts.append(f"- {item}\n")
            parts.append("\n")

        # Add question
        parts.append(f"Question: {sample.question}\n")

        # Add choices if available
        if hasattr(sample, "choices") and sample.choices:
            parts.append("\nChoices:")
            for key in sorted(sample.choices.keys()):
                parts.append(f"\n{key}. {sample.choices[key]}")
            parts.append("\n\nRespond with only the choice letter (A, B, C, or D).")

        return "".join(parts)
