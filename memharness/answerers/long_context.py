"""Long context baseline answerer."""

from __future__ import annotations

from httpx import AsyncClient, Limits
from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import increment_eval_metric

# HTTP client configuration for high concurrency
_MAX_CONNECTIONS = 20
_MAX_KEEPALIVE_CONNECTIONS = 50
_REQUEST_TIMEOUT_SECONDS = 300.0


def long_context_answerer(
    model: str,
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
):
    """Factory that returns async task callable for pydantic-evals.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-opus-4-20250514")
        provider: Provider name ("openai", "anthropic")
        base_url: Optional base URL for custom endpoints
        api_key: Optional API key

    Returns:
        Async task function (inputs: dict) -> str
    """
    _agent = None  # Lazy initialization via closure

    def get_agent() -> Agent:
        nonlocal _agent
        if _agent is not None:
            return _agent

        http_client = AsyncClient(
            limits=Limits(
                max_connections=_MAX_CONNECTIONS,
                max_keepalive_connections=_MAX_KEEPALIVE_CONNECTIONS,
            ),
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

        if provider == "anthropic":
            model_obj = AnthropicModel(model)
        elif provider == "openai":
            model_obj = OpenAIChatModel(
                model,
                provider=OpenAIProvider(
                    base_url=base_url,
                    api_key=api_key,
                    http_client=http_client,
                ),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        _agent = Agent(model=model_obj)
        return _agent

    async def task(inputs: dict) -> str:
        """Task function called by pydantic-evals.

        Args:
            inputs: Dict with conversation, question, choices

        Returns:
            Response string (choice letter for multiple choice)
        """
        agent = get_agent()

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

        return str(result.output)

    return task
