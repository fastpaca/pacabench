"""Long context baseline answerer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# Types colocated with answerer
@dataclass
class AnswerResult:
    """Result from an answerer."""

    response: str  # The actual response (choice letter, free text, etc.)
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    metrics: dict[str, Any] = field(default_factory=dict)


class ChoiceOutput(BaseModel):
    """Structured output for multiple choice questions."""

    choice: str
    reasoning: str | None = None


class LongContextAnswerer:
    """Baseline answerer that uses full conversation context.

    This answerer provides the complete conversation history to the LLM
    without any memory system or context management.
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize long context answerer.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-opus-4-20250514")
            provider: Provider name ("openai", "anthropic")
            base_url: Optional base URL for custom endpoints
            api_key: Optional API key
            system_prompt: Optional system prompt override
        """
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._agent = None  # Lazy initialization

    @property
    def agent(self) -> Agent:
        """Lazy-load the agent."""
        if self._agent is None:
            if self.provider == "anthropic":
                model = AnthropicModel(self.model, api_key=self.api_key)
            elif self.provider == "openai":
                model = OpenAIChatModel(
                    self.model,
                    provider=OpenAIProvider(
                        base_url=self.base_url,
                        api_key=self.api_key,
                    ),
                )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            self._agent = Agent(model=model)
        return self._agent

    async def answer(self, sample: Any) -> AnswerResult:
        """Generate an answer for the given sample.

        Args:
            sample: Dataset sample with conversation, question, and choices

        Returns:
            AnswerResult with response and metrics
        """
        start_time = time.time()

        # Format prompt
        prompt = self._format_prompt(sample)

        # Run agent asynchronously
        result = await self.agent.run(prompt)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Extract response - result.output contains the actual data
        if hasattr(result.output, "choice"):
            response = result.output.choice
        else:
            response = str(result.output)

        # Get token usage from result.usage()
        usage = result.usage()

        return AnswerResult(
            response=response,
            total_latency_ms=latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    def _format_prompt(self, sample: Any) -> str:
        """Format the sample into a prompt."""
        parts = []

        # Add conversation history
        if hasattr(sample, "conversation") and sample.conversation:
            parts.append("Prior conversation:\n")
            for turn in sample.conversation:
                role = turn["role"].capitalize()
                content = turn["content"]
                parts.append(f"{role}: {content}\n")

        # Add question
        parts.append(f"\nQuestion: {sample.question}\n")

        # Add choices if available (for multiple choice)
        if hasattr(sample, "choices") and sample.choices:
            parts.append("\nChoices:")
            for key in sorted(sample.choices.keys()):
                parts.append(f"\n{key}. {sample.choices[key]}")

            parts.append("\n\nRespond with only the choice letter (A, B, C, or D).")

        return "".join(parts)

    def _default_system_prompt(self) -> str:
        """Default system prompt for the answerer."""
        return (
            "You are answering questions about conversations. "
            "Provide accurate, concise answers based on the information given."
        )
