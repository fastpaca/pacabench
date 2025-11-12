"""Q&A executor for long-context baseline."""

from __future__ import annotations

from httpx import AsyncClient, Limits
from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import increment_eval_metric

from memharness.executors.base import Executor

_MAX_CONNECTIONS = 20
_MAX_KEEPALIVE_CONNECTIONS = 50
_REQUEST_TIMEOUT_SECONDS = 300.0


class QAExecutor(Executor):
    """Long-context Q&A baseline executor.

    This executor handles Q&A tasks with conversation history, using the full
    context window without external memory systems. It works for datasets like
    MemBench and LongMemEval where the task is to answer questions based on
    conversation history.
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """Initialize Q&A executor.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
            provider: Provider name ("openai", "anthropic")
            base_url: Optional base URL for custom endpoints
            api_key: Optional API key
            **kwargs: Additional configuration
        """
        super().__init__(model, provider, **kwargs)
        self.base_url = base_url
        self.api_key = api_key
        self._agent = None

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
        """Execute Q&A task.

        Args:
            inputs: Dict with:
                - conversation: List of message turns
                - question: Question to answer
                - choices: Optional multiple choice options

        Returns:
            Response string (choice letter for multiple choice, or answer text)
        """
        agent = self._get_agent()

        message_history = []
        for turn in inputs.get("conversation", []):
            if turn["role"] == "user":
                message_history.append(
                    ModelRequest(parts=[UserPromptPart(content=turn["content"])])
                )
            elif turn["role"] == "assistant":
                message_history.append(ModelResponse(parts=[TextPart(content=turn["content"])]))

        prompt = f"Question: {inputs['question']}"

        if "choices" in inputs:
            choices_text = "\n".join(
                f"{key}. {value}" for key, value in sorted(inputs["choices"].items())
            )
            prompt += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

        result = await agent.run(prompt, message_history=message_history)

        usage = result.usage()
        increment_eval_metric("input_tokens", usage.input_tokens)
        increment_eval_metric("output_tokens", usage.output_tokens)
        increment_eval_metric("cache_write_tokens", usage.cache_write_tokens)
        increment_eval_metric("cache_read_tokens", usage.cache_read_tokens)

        return str(result.output)
