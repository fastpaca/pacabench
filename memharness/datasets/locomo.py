"""Locomo dataset implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel


@dataclass
class LocomoSample:
    """A single Locomo sample."""

    id: str
    conversation: list[dict[str, str]]
    question: str
    ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""

    correct: bool
    sample_id: str
    response: str
    ground_truth: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    metrics: dict[str, Any] = field(default_factory=dict)


class LocomoDataset:
    """Locomo dataset for long context memory evaluation.

    Uses LLM-as-a-judge for evaluation.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        judge_model: str = "claude-3-5-sonnet-20241022",
    ):
        """Initialize Locomo dataset.

        Args:
            data_dir: Path to Locomo data directory
            judge_model: Model to use for LLM-as-judge evaluation
        """
        if data_dir is None:
            # TODO: Download from HuggingFace on first use
            data_dir = Path(__file__).parent.parent.parent / "data" / "locomo"

        self.data_dir = Path(data_dir)
        self.judge_model = judge_model
        self._judge = None  # Lazy initialization

    @property
    def judge(self) -> Agent:
        """Lazy-load the judge agent."""
        if self._judge is None:
            self._judge = Agent(model=AnthropicModel(self.judge_model))
        return self._judge

    def load(self, limit: int | None = None) -> list[LocomoSample]:
        """Load samples from the dataset.

        Args:
            limit: Optional limit on number of samples to return

        Returns:
            List of LocomoSample objects
        """
        # TODO: Implement Locomo data loading
        # For now, return empty list as stub
        raise NotImplementedError(
            "Locomo dataset not yet implemented. "
            "Please add implementation or download from HuggingFace."
        )

    def evaluate(self, sample: LocomoSample, answer_result: Any) -> EvalResult:
        """Evaluate an answerer's response using LLM-as-judge.

        Args:
            sample: The Locomo sample
            answer_result: AnswerResult from answerer

        Returns:
            EvalResult with correctness and metrics
        """
        # Use LLM judge to evaluate correctness
        judge_prompt = f"""You are evaluating a model's answer to a question about a conversation.

Question: {sample.question}

Ground Truth Answer: {sample.ground_truth}

Model Answer: {answer_result.response}

Is the model's answer correct? Respond with exactly "CORRECT" or "WRONG" and provide brief reasoning."""

        judge_result = self.judge.run_sync(judge_prompt)
        judgment = judge_result.data.strip().upper()

        # Parse judgment
        correct = "CORRECT" in judgment

        return EvalResult(
            correct=correct,
            sample_id=sample.id,
            response=answer_result.response,
            ground_truth=sample.ground_truth,
            latency_ms=answer_result.total_latency_ms,
            input_tokens=answer_result.input_tokens,
            output_tokens=answer_result.output_tokens,
            metrics={
                **answer_result.metrics,
                "judge_response": judge_result.data,
                "judge_tokens": (
                    judge_result.usage().request_tokens
                    + judge_result.usage().response_tokens
                ),
            },
        )
