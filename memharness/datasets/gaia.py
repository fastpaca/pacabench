"""GAIA (General AI Assistants) dataset implementation.

GAIA is a benchmark for testing AI assistants on real-world questions
requiring reasoning, multi-modality, web browsing, and tool use.

Dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
Paper: https://arxiv.org/abs/2311.12983
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import load_dataset
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_evals import Case, Dataset, increment_eval_metric
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from memharness.datasets.base import BenchmarkDataset
from memharness.executors.gaia import GAIAExecutor

if TYPE_CHECKING:
    from memharness.executors.base import Executor

_HF_REPO_ID = "gaia-benchmark/GAIA"
_LEVELS = {
    "level1": "2023_level1",
    "level2": "2023_level2",
    "level3": "2023_level3",
    "all": "2023_all",
}


class GAIAEvaluator(Evaluator):
    """Evaluator using LLM-as-judge for GAIA answers.

    GAIA answers can be complex and require semantic understanding,
    so we use an LLM judge to compare responses.
    """

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
        """Synchronous wrapper for async evaluation."""
        import asyncio

        return asyncio.run(self.evaluate_async(ctx))

    async def evaluate_async(self, ctx: EvaluatorContext) -> bool:
        """Async evaluation using LLM judge."""
        response = ctx.output.strip() if ctx.output else ""
        expected = ctx.expected_output.strip() if ctx.expected_output else ""

        if not expected or not response:
            return False

        prompt = f"""You are evaluating if an AI assistant's answer matches the expected answer for a GAIA benchmark question.

Question: {ctx.inputs.get("question", "N/A")}

Expected Answer: {expected}

Assistant's Answer: {response}

Does the assistant's answer match the expected answer? Consider:
- The core factual content must be correct
- Paraphrasing is acceptable
- Extra context/explanation is acceptable if core answer is present
- Minor formatting differences are acceptable (e.g., "50%" vs "50 percent")
- But factual errors or missing key information mean NO match

Respond with ONLY "YES" or "NO"."""

        agent = self._get_agent()
        result = await agent.run(prompt)
        judgment = str(result.output).strip().upper()

        usage = result.usage()
        increment_eval_metric("judge_input_tokens", usage.input_tokens)
        increment_eval_metric("judge_output_tokens", usage.output_tokens)

        return judgment.startswith("YES")


class GAIA(BenchmarkDataset):
    """GAIA benchmark dataset.

    Real-world questions requiring reasoning, multi-modality, web browsing,
    and tool use. Questions are divided into 3 difficulty levels.

    Uses GAIAExecutor by default which provides agentic capabilities via
    smolagents (web search, code execution, multi-step reasoning).
    """

    name = "gaia"
    default_executor = GAIAExecutor

    def __init__(
        self,
        level: str = "all",
        split: str = "validation",
        judge_model: str = "gpt-4o-mini",
        executor_class: type[Executor] | None = None,
    ):
        """Initialize GAIA dataset.

        Args:
            level: Difficulty level ("level1", "level2", "level3", or "all")
            split: Dataset split ("validation" or "test")
            judge_model: Model to use for LLM-as-judge evaluation
            executor_class: Optional executor class override
        """
        super().__init__(executor_class=executor_class)
        if level not in _LEVELS:
            raise ValueError(f"Invalid level '{level}'. Available: {list(_LEVELS.keys())}")
        self.level = level
        self.split = split
        self.judge_model = judge_model

    def load(self, limit: int | None = None) -> Dataset:
        """Load GAIA dataset from HuggingFace.

        Args:
            limit: Optional limit on number of cases to load

        Returns:
            pydantic-evals Dataset with cases and evaluators
        """
        # Load from HuggingFace
        # Note: GAIA is a gated dataset. Users must:
        # 1. Accept terms at https://huggingface.co/datasets/gaia-benchmark/GAIA
        # 2. Authenticate with `huggingface-cli login`
        dataset_name = _LEVELS[self.level]
        hf_dataset = load_dataset(_HF_REPO_ID, dataset_name, split=self.split)

        # Apply limit if specified
        if limit is not None:
            hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

        # Convert to pydantic-evals Cases
        cases = []
        for item in hf_dataset:
            # Build inputs dict
            inputs: dict[str, Any] = {
                "question": item["Question"],
            }

            # Add file context if present
            if item.get("file_name"):
                inputs["file_name"] = item["file_name"]
                inputs["file_path"] = item.get("file_path", "")

            # Add annotator metadata
            if item.get("Annotator Metadata"):
                inputs["metadata"] = item["Annotator Metadata"]

            cases.append(
                Case(
                    name=item.get("task_id", f"gaia_{len(cases)}"),
                    inputs=inputs,
                    expected_output=item["Final answer"],
                    metadata={
                        "level": item.get("Level", ""),
                        "file_name": item.get("file_name", ""),
                        "annotator_metadata": item.get("Annotator Metadata", {}),
                    },
                )
            )

        return Dataset(
            cases=cases,
            evaluators=[GAIAEvaluator(model=self.judge_model)],
        )


def load_gaia(
    level: str = "all",
    split: str = "validation",
    limit: int | None = None,
    judge_model: str = "gpt-4o-mini",
) -> Dataset:
    """Load GAIA dataset as pydantic-evals Dataset.

    Args:
        level: Difficulty level ("level1", "level2", "level3", or "all")
        split: Dataset split ("validation" or "test")
        limit: Optional limit on number of samples
        judge_model: Model to use for LLM-as-judge evaluation

    Returns:
        pydantic-evals Dataset with Cases
    """
    gaia = GAIA(level=level, split=split, judge_model=judge_model)
    return gaia.load(limit=limit)
