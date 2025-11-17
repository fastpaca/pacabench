"""GAIA dataset."""

from collections.abc import Iterable
from typing import Any

from datasets import load_dataset
from openai import AsyncOpenAI

from agentbench.datasets.base import Dataset
from agentbench.types import Case, EvaluationResult


class GaiaDataset(Dataset):
    """GAIA agentic dataset."""

    level: str = "all"

    async def load(self, limit: int | None = None) -> Iterable[Case]:
        """Load GAIA cases."""
        levels = {
            "level1": "2023_level1",
            "level2": "2023_level2",
            "level3": "2023_level3",
            "all": "2023_all",
        }

        if self.level not in levels:
            raise ValueError(f"Invalid level '{self.level}'. Available: {list(levels.keys())}")

        dataset_name = levels[self.level]
        split = self.split or "validation"
        hf_dataset = load_dataset("gaia-benchmark/GAIA", dataset_name, split=split)

        if limit is not None:
            hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

        cases = []
        for item in hf_dataset:
            inputs: dict[str, Any] = {
                "question": item["Question"],
            }

            if item.get("file_name"):
                inputs["file_name"] = item["file_name"]
                inputs["file_path"] = item.get("file_path", "")

            if item.get("Annotator Metadata"):
                inputs["metadata"] = item["Annotator Metadata"]

            cases.append(
                Case(
                    id=item.get("task_id", f"gaia_{len(cases)}"),
                    task_type="agentic",
                    inputs=inputs,
                    expected_output=item["Final answer"],
                    metadata={
                        "level": item.get("Level", ""),
                        "file_name": item.get("file_name", ""),
                    },
                )
            )

        return cases

    async def eval(
        self,
        case: Case,
        output: str | None,
        error: str | None,
        judge_model: str = "gpt-4o-mini",
        judge_client: AsyncOpenAI | None = None,
    ) -> tuple[EvaluationResult, dict[str, int] | None]:
        """Evaluate GAIA case using LLM-as-judge."""
        if error or not output:
            return EvaluationResult(passed=False, judge_passed=False), None

        judge_passed, judge_metrics = await self._evaluate_gaia(
            case, output, judge_model, judge_client
        )
        return (
            EvaluationResult(passed=judge_passed, judge_passed=judge_passed),
            judge_metrics,
        )

    async def _evaluate_gaia(
        self,
        case: Case,
        output: str,
        model: str,
        judge_client: AsyncOpenAI | None,
    ) -> tuple[bool, dict[str, int]]:
        """Evaluate GAIA answer using LLM-as-judge."""
        expected = case.expected_output
        if not output or not expected:
            return False, {"input_tokens": 0, "output_tokens": 0}

        response = output.strip()
        expected_text = expected.strip()
        question = case.inputs.get("question", "N/A")

        prompt = f"""You are evaluating if an AI assistant's answer matches the expected answer for a GAIA benchmark question.

Question: {question}

Expected Answer: {expected_text}

Assistant's Answer: {response}

Does the assistant's answer match the expected answer? Consider:
- The core factual content must be correct
- Paraphrasing is acceptable
- Extra context/explanation is acceptable if core answer is present
- Minor formatting differences are acceptable (e.g., "50%" vs "50 percent")
- But factual errors or missing key information mean NO match

Respond with ONLY "YES" or "NO"."""

        if judge_client is None:
            judge_client = AsyncOpenAI()

        completion = await judge_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        judgment = completion.choices[0].message.content or ""
        judgment = judgment.strip().upper()

        usage = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
        }

        judge_passed = judgment.startswith("YES")
        return judge_passed, usage
