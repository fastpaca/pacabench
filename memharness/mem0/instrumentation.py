"""Instrumentation hooks to attribute mem0 internal token and cost usage."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any

from genai_prices import Usage, calc_price
from pydantic_evals import increment_eval_metric

_PATCHED = False


def patch_mem0_instrumentation() -> None:
    """Monkeypatch mem0 OpenAI providers to track token/cost usage."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from mem0.embeddings.openai import OpenAIEmbedding  # type: ignore
        from mem0.llms.openai import OpenAILLM  # type: ignore
    except Exception:
        # If mem0 or OpenAI providers are unavailable (e.g., tests without dependency),
        # gracefully skip instrumentation.
        return

    _patch_openai_embedding(OpenAIEmbedding)
    _patch_openai_llm(OpenAILLM)
    _PATCHED = True


def _patch_openai_embedding(cls: type) -> None:
    if getattr(cls.embed, "_memharness_patched", False):
        return

    def instrumented_embed(self, text: str, memory_action: str | None = None):
        sanitized = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[sanitized],
            model=self.config.model,
            dimensions=self.config.embedding_dims,
        )
        usage = getattr(response, "usage", None)
        _record_embedding_usage(self.config.model, usage)
        return response.data[0].embedding

    instrumented_embed._memharness_patched = True  # type: ignore[attr-defined]
    cls.embed = instrumented_embed  # type: ignore[assignment]


def _patch_openai_llm(cls: type) -> None:
    if getattr(cls.generate_response, "_memharness_patched", False):
        return

    def instrumented_generate_response(
        self,
        messages: list[dict[str, str]],
        response_format: Any = None,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update({"model": self.config.model, "messages": messages})

        provider_id: str | None = None
        if os.getenv("OPENROUTER_API_KEY"):
            provider_id = "openrouter"
            openrouter_params: dict[str, Any] = {}
            if self.config.models:
                openrouter_params["models"] = self.config.models
                openrouter_params["route"] = self.config.route
                params.pop("model", None)
            if self.config.site_url and self.config.app_name:
                openrouter_params["extra_headers"] = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
            params.update(**openrouter_params)
        else:
            provider_id = "openai"
            for param in ["store"]:
                if hasattr(self.config, param):
                    params[param] = getattr(self.config, param)

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        usage = getattr(response, "usage", None)
        _record_llm_usage(self.config.model, provider_id, usage)

        parsed_response = self._parse_response(response, tools)
        if self.config.response_callback:
            with suppress(Exception):
                self.config.response_callback(self, response, params)
        return parsed_response

    instrumented_generate_response._memharness_patched = True  # type: ignore[attr-defined]
    cls.generate_response = instrumented_generate_response  # type: ignore[assignment]


def _safe_increment(metric: str, value: float | int | None) -> None:
    if value is None:
        return
    with suppress(Exception):
        increment_eval_metric(metric, value)


def _record_cost(
    model: str, provider_id: str | None, input_tokens: int, output_tokens: int
) -> float:
    if input_tokens <= 0 and output_tokens <= 0:
        return 0.0

    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=0,
        cache_read_tokens=0,
    )
    price = calc_price(usage, model_ref=model, provider_id=provider_id)
    cost = float(price.total_price) if price else 0.0
    if cost > 0:
        _safe_increment("mem0_internal_cost_usd", cost)
    return cost


def _record_embedding_usage(model: str, usage: Any) -> None:
    if usage is None:
        return
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = getattr(usage, "prompt_tokens", None)
    if total_tokens is None:
        return
    total_tokens = int(total_tokens)

    _safe_increment("mem0_embedding_tokens", total_tokens)
    _record_cost(model, "openai", total_tokens, 0)


def _record_llm_usage(model: str, provider_id: str | None, usage: Any) -> None:
    if usage is None:
        return
    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", None)

    output_tokens = getattr(usage, "completion_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "output_tokens", None)

    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)

    if input_tokens is not None:
        _safe_increment("mem0_llm_input_tokens", int(input_tokens))
    if output_tokens is not None:
        _safe_increment("mem0_llm_output_tokens", int(output_tokens))
    if total_tokens is not None:
        _safe_increment("mem0_llm_total_tokens", int(total_tokens))

    _record_cost(
        model,
        provider_id,
        int(input_tokens or total_tokens or 0),
        int(output_tokens or 0),
    )
