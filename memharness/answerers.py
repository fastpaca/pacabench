from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .types import AnswerResult, QAInput, UsageStats


class Answerer(ABC):
    """Abstract base class for pluggable QA answerers."""

    @abstractmethod
    def answer(self, qa_input: QAInput) -> AnswerResult:
        """Return a choice for the provided QAInput."""


class PydanticAgentAnswerer(Answerer):
    """
    Generic Answerer wrapper around a pydantic-ai Agent.

    The agent must accept a single string prompt and return a result whose
    `.output` has a `choice` attribute. This matches the behavior of
    agents configured with `output_type=ChoiceResult`.
    """

    def __init__(self, agent, prompt_renderer=None):
        self.agent = agent
        self.prompt_renderer = prompt_renderer or default_prompt_renderer

    def answer(self, qa_input: QAInput) -> AnswerResult:
        prompt = self.prompt_renderer(qa_input)
        generation = self.agent.run_sync(prompt)
        usage = generation.usage()
        usage_stats = None
        if usage:
            usage_stats = UsageStats(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                requests=getattr(usage, "requests", 1) or 1,
            )
        return AnswerResult(
            choice=getattr(generation.output, "choice", None),
            response_text=getattr(generation.output, "choice", None),
            usage=usage_stats,
            extra={"prompt": prompt},
        )


def default_prompt_renderer(qa_input: QAInput) -> str:
    conversation_lines: List[str] = []
    for turn in qa_input.history:
        role = turn.get("role", "user")
        speaker = "User" if role == "user" else "Assistant"
        conversation_lines.append(f"{speaker}: {turn.get('content', '').strip()}")
    conversation_block = "\n".join(conversation_lines)
    choice_lines = [f"{label}. {text}" for label, text in sorted(qa_input.choices.items())]
    choice_block = "\n".join(choice_lines)
    timestamp = qa_input.timestamp or "unknown time"
    source = qa_input.metadata.get("source_file", "unknown")
    return f"""
Conversation source: {source}
Prior conversation:
{conversation_block}

Question asked at {timestamp}:
{qa_input.question}

Choices:
{choice_block}

Return ONLY the JSON object with the single best choice.
""".strip()


def build_openai_agent(model_name: str, base_url: str, api_key: Optional[str]):
    from pydantic import BaseModel, Field, field_validator
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    class ChoiceResult(BaseModel):
        choice: str = Field(description="The single best answer option.")

        @field_validator("choice")
        @classmethod
        def ensure_choice(cls, value: str) -> str:
            normalized = value.strip().upper()
            if normalized not in {"A", "B", "C", "D"}:
                raise ValueError(f"Choice must be one of ['A', 'B', 'C', 'D'], got {value!r}")
            return normalized

    provider = OpenAIProvider(base_url=base_url, api_key=api_key or "lm-studio")
    model = OpenAIChatModel(model_name, provider=provider)
    system_prompt = """You are an expert memory QA assistant.
You will receive the full conversation history that the user previously had with you.
You must pick the single best option (A, B, C, or D) that answers the provided question.
Respond ONLY with a JSON object of the shape {"choice": "<letter>"}.
"""
    return Agent(model=model, output_type=ChoiceResult, system_prompt=system_prompt)


@dataclass
class AnswererSpec:
    name: str
    description: str
    builder: Callable[..., Answerer]
    default_params: Dict[str, Any] = field(default_factory=dict)


def build_pydantic_agent_answerer(
    provider: str = "openai",
    model: str = "lm-studio",
    base_url: str = "http://127.0.0.1:1234/v1",
    api_key: Optional[str] = None,
) -> Answerer:
    provider = provider.lower()
    if provider == "openai":
        agent = build_openai_agent(model, base_url, api_key)
    else:
        raise ValueError(f"Unsupported provider '{provider}' for pydantic-agent answerer.")
    return PydanticAgentAnswerer(agent, prompt_renderer=default_prompt_renderer)


ANSWERER_REGISTRY: Dict[str, AnswererSpec] = {
    "pydantic-agent": AnswererSpec(
        name="pydantic-agent",
        description="Generic PydanticAI agent answerer (provider switch).",
        builder=build_pydantic_agent_answerer,
    ),
}


def list_answerers() -> List[AnswererSpec]:
    return list(ANSWERER_REGISTRY.values())


def get_answerer_spec(name: str) -> AnswererSpec:
    try:
        return ANSWERER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown answerer '{name}'. Available: {list(ANSWERER_REGISTRY)}") from exc


def build_answerer(name: str, **kwargs) -> Answerer:
    spec = get_answerer_spec(name)
    params = {**spec.default_params, **{k: v for k, v in kwargs.items() if v is not None}}
    return spec.builder(**params)


__all__ = [
    "Answerer",
    "PydanticAgentAnswerer",
    "default_prompt_renderer",
    "build_answerer",
    "list_answerers",
    "get_answerer_spec",
]
