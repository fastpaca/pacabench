"""LongMemEval dataset."""

import json
from collections.abc import Iterable

import tiktoken
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

from agentbench.datasets.base import Dataset
from agentbench.stages.case import Case
from agentbench.stages.result import CaseResult


def _flatten_sessions(sessions: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    """Flatten nested session structure into a single conversation history."""
    turns = []

    for session in sessions:
        if isinstance(session, list):
            turns.extend(session)
        elif isinstance(session, dict):
            turns.append(session)

    return [turn for turn in turns if isinstance(turn, dict) and turn.get("content")]


class LongMemEvalDataset(Dataset):
    """LongMemEval QA dataset."""

    async def load(self, limit: int | None = None) -> Iterable[Case]:
        """Load LongMemEval cases."""
        split = self.split or "s_cleaned"
        split_files = {
            "s_cleaned": "longmemeval_s_cleaned.json",
            "m_cleaned": "longmemeval_m_cleaned.json",
        }

        if split not in split_files:
            raise ValueError(f"Invalid split '{split}'. Available: {list(split_files.keys())}")

        file_path = hf_hub_download(
            repo_id="xiaowu0162/longmemeval-cleaned",
            filename=split_files[split],
            repo_type="dataset",
        )

        with open(file_path, encoding="utf-8") as f:
            raw_data = json.load(f)

        if limit is not None:
            raw_data = raw_data[:limit]

        cases = []
        for sample in raw_data:
            conversation = _flatten_sessions(sample["haystack_sessions"])
            cases.append(
                Case(
                    id=sample["question_id"],
                    task_type="qa",
                    inputs={
                        "conversation": conversation,
                        "question": sample["question"],
                    },
                    expected_output=sample["answer"],
                    metadata={
                        "question_type": sample["question_type"],
                        "question_date": sample["question_date"],
                        "answer_session_ids": sample["answer_session_ids"],
                    },
                )
            )

        return cases

    async def eval(
        self,
        case: Case,
        result: CaseResult,
        judge_model: str = "gpt-4o-mini",
        judge_client: AsyncOpenAI | None = None,
    ) -> CaseResult:
        """Evaluate LongMemEval case using F1 and LLM judge."""
        if result.error or not result.output:
            return result.model_copy(update={"passed": False})

        f1_score, f1_passed = self._evaluate_f1_score(case, result.output)
        judge_passed, judge_metrics = await self._evaluate_llm_judge(
            case, result.output, judge_model, judge_client
        )
        return result.model_copy(
            update={
                "passed": f1_passed and judge_passed,
                "f1_score": f1_score,
                "f1_passed": f1_passed,
                "judge_passed": judge_passed,
                "judge_metrics": judge_metrics,
            }
        )

    def _evaluate_f1_score(self, case: Case, output: str) -> tuple[float, bool]:
        """Evaluate using F1 score based on token overlap."""
        expected = case.expected_output
        if not output or not expected:
            return 0.0, False

        encoding = tiktoken.get_encoding("cl100k_base")
        response_tokens = set(encoding.encode(output.strip().lower()))
        expected_tokens = set(encoding.encode(expected.strip().lower()))

        if not expected_tokens:
            return 0.0, False

        overlap = response_tokens & expected_tokens
        if not overlap:
            return 0.0, False

        precision = len(overlap) / len(response_tokens) if response_tokens else 0.0
        recall = len(overlap) / len(expected_tokens) if expected_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1, f1 > 0.5

    async def _evaluate_llm_judge(
        self,
        case: Case,
        output: str,
        model: str,
        judge_client: AsyncOpenAI | None,
    ) -> tuple[bool, dict[str, int]]:
        """Evaluate using LLM-as-judge for semantic equivalence."""
        expected = case.expected_output
        if not output or not expected:
            return False, {"input_tokens": 0, "output_tokens": 0}

        response = output.strip()
        expected_text = expected.strip()
        question = case.inputs.get("question", "N/A")

        prompt = f"""You are evaluating if a model's answer is semantically equivalent to the expected answer.

Question: {question}

Expected Answer: {expected_text}

Model's Answer: {response}

Does the model's answer convey the same information as the expected answer? Consider:
- Paraphrasing is acceptable
- Extra explanation is acceptable if core answer is present
- Minor details can differ if main point matches

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
