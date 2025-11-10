"""LongMemEval dataset implementation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_evals import Case, Dataset, increment_eval_metric
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

_HF_REPO_ID = "xiaowu0162/longmemeval-cleaned"
_SPLIT_FILES = {
    "s_cleaned": "longmemeval_s_cleaned.json",
    "m_cleaned": "longmemeval_m_cleaned.json",
}


class F1ScoreEvaluator(Evaluator):
    """Evaluator that computes F1 score based on token overlap."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        """Compute F1 score and store as metric, return True if F1 > 0.5."""
        response = ctx.output.strip().lower() if ctx.output else ""
        expected = ctx.expected_output.strip().lower() if ctx.expected_output else ""

        if not expected or not response:
            increment_eval_metric("f1_score", 0.0)
            return False

        response_tokens = set(self._tokenize(response))
        expected_tokens = set(self._tokenize(expected))

        if not expected_tokens:
            increment_eval_metric("f1_score", 0.0)
            return False

        overlap = response_tokens & expected_tokens
        if not overlap:
            increment_eval_metric("f1_score", 0.0)
            return False

        precision = len(overlap) / len(response_tokens) if response_tokens else 0.0
        recall = len(overlap) / len(expected_tokens) if expected_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        increment_eval_metric("f1_score", f1)
        return f1 > 0.5

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenization with punctuation handling."""
        return [token for token in re.findall(r"\b\w+\b", text.lower()) if token]


class LLMJudgeEvaluator(Evaluator):
    """Evaluator using LLM-as-judge for semantic equivalence."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize with judge model."""
        self.model = model
        self._agent: Agent | None = None

    def _get_agent(self) -> Agent:
        """Lazy initialize judge agent."""
        if self._agent is None:
            self._agent = Agent(OpenAIChatModel(self.model))
        return self._agent

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        """Synchronous wrapper (should not be called in async context)."""
        import asyncio

        return asyncio.run(self.evaluate_async(ctx))

    async def evaluate_async(self, ctx: EvaluatorContext) -> bool:
        """Async evaluation using LLM judge."""
        response = ctx.output.strip() if ctx.output else ""
        expected = ctx.expected_output.strip() if ctx.expected_output else ""

        if not expected or not response:
            return False

        prompt = f"""You are evaluating if a model's answer is semantically equivalent to the expected answer.

Question: {ctx.inputs.get("question", "N/A")}

Expected Answer: {expected}

Model's Answer: {response}

Does the model's answer convey the same information as the expected answer? Consider:
- Paraphrasing is acceptable
- Extra explanation is acceptable if core answer is present
- Minor details can differ if main point matches

Respond with ONLY "YES" or "NO"."""

        agent = self._get_agent()
        result = await agent.run(prompt)
        judgment = str(result.output).strip().upper()

        usage = result.usage()
        increment_eval_metric("judge_input_tokens", usage.input_tokens)
        increment_eval_metric("judge_output_tokens", usage.output_tokens)

        return judgment.startswith("YES")


def load_longmemeval(
    split: str = "s_cleaned",
    limit: int | None = None,
    cache_dir: Path | str | None = None,
    judge_model: str = "gpt-4o-mini",
) -> Dataset:
    """Load LongMemEval dataset as pydantic-evals Dataset.

    Args:
        split: "s_cleaned" (single-session) or "m_cleaned" (multi-session)
        limit: Optional limit on number of samples
        cache_dir: Optional cache directory for HuggingFace downloads
        judge_model: Model to use for LLM-as-judge evaluation

    Returns:
        pydantic-evals Dataset with Cases
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"Invalid split '{split}'. Available splits: {list(_SPLIT_FILES.keys())}")

    file_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_SPLIT_FILES[split],
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    with open(file_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    if limit is not None:
        raw_data = raw_data[:limit]

    cases = [_build_case(sample) for sample in raw_data]

    return Dataset(
        cases=cases,
        evaluators=[
            F1ScoreEvaluator(),
            LLMJudgeEvaluator(model=judge_model),
        ],
    )


def _build_case(sample: dict[str, Any]) -> Case:
    """Build a pydantic-evals Case from a raw sample."""
    conversation_history = _flatten_sessions(sample["haystack_sessions"])

    return Case(
        name=sample["question_id"],
        inputs={
            "conversation": conversation_history,
            "question": sample["question"],
        },
        expected_output=sample["answer"],
        metadata={
            "question_type": sample["question_type"],
            "question_date": sample["question_date"],
            "answer_session_ids": sample["answer_session_ids"],
            "haystack_dates": sample["haystack_dates"],
            "haystack_session_ids": sample["haystack_session_ids"],
        },
    )


def _flatten_sessions(sessions: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    """Flatten nested session structure into a single conversation history."""
    turns = []

    for session in sessions:
        if isinstance(session, list):
            turns.extend(session)
        elif isinstance(session, dict):
            turns.append(session)

    return [turn for turn in turns if isinstance(turn, dict) and turn.get("content")]
