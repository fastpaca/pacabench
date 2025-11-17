"""MemBench dataset."""

import json
from collections.abc import Iterable
from pathlib import Path

import tiktoken
from git import Repo
from loguru import logger
from openai import AsyncOpenAI

from agentbench.datasets.base import Dataset
from agentbench.types import Case, EvaluationResult

_MEMBENCH_GITHUB_REPO_URL = "https://github.com/import-myself/Membench.git"
_MEMBENCH_GITHUB_BRANCH = "main"


def _to_conversation(message_list: list) -> list[dict[str, str]]:
    """Convert message_list to a flat conversation."""
    turns = []
    for block in message_list:
        items = block if isinstance(block, list) else [block]
        for item in items:
            if isinstance(item, dict):
                meta_parts = []
                if item.get("time"):
                    meta_parts.append(f"time: {item['time']}")
                if item.get("place"):
                    meta_parts.append(f"place: {item['place']}")
                meta_suffix = f" ({'; '.join(meta_parts)})" if meta_parts else ""

                user_text = item.get("user_message") or item.get("user")
                assistant_text = item.get("assistant_message") or item.get("assistant")

                if user_text:
                    turns.append({"role": "user", "content": f"{user_text}{meta_suffix}"})
                if assistant_text:
                    turns.append({"role": "assistant", "content": f"{assistant_text}{meta_suffix}"})
            elif item:
                turns.append({"role": "user", "content": str(item)})
    return [turn for turn in turns if turn.get("content")]


def _normalize_qa_value(value: str | list[str] | None) -> str:
    """Normalize QA field value (string or list) to string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _normalize_choices(choices: dict | None) -> dict[str, str]:
    """Normalize choices dict values to strings."""
    if not choices:
        return {}
    return {k: _normalize_qa_value(v) for k, v in choices.items()}


def _record_to_case(record: dict, dataset_name: str) -> Case:
    """Convert a MemBench record dict to a Case."""
    tid = record["tid"]
    sample_id = f"{dataset_name}::{tid}"
    conversation = _to_conversation(record["message_list"])

    qa = record["QA"]
    question = qa.get("question", "")
    choices = _normalize_choices(qa.get("choices"))
    answer = _normalize_qa_value(qa.get("answer"))
    ground_truth = _normalize_qa_value(qa.get("ground_truth"))

    return Case(
        id=sample_id,
        task_type="qa",
        inputs={
            "conversation": conversation,
            "question": question,
            "choices": choices,
        },
        expected_output=ground_truth or answer,
        metadata={
            "source_file": dataset_name,
            "tid": tid,
        },
    )


def _ensure_membench_repo(repo_dir: Path) -> None:
    """Clone or update MemBench repository from GitHub."""
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if repo_dir.exists() and (repo_dir / ".git").exists():
        try:
            repo = Repo(repo_dir)
            repo.remotes.origin.fetch()
            repo.git.checkout(_MEMBENCH_GITHUB_BRANCH)
            repo.git.pull()
        except Exception as e:
            raise RuntimeError(f"Failed to update MemBench repository: {e}") from e
    else:
        try:
            Repo.clone_from(_MEMBENCH_GITHUB_REPO_URL, repo_dir, branch=_MEMBENCH_GITHUB_BRANCH)
        except Exception as e:
            raise RuntimeError(f"Failed to clone MemBench repository: {e}") from e


class MemBenchDataset(Dataset):
    """MemBench QA dataset."""

    agent_type: str = "FirstAgent"

    async def load(self, limit: int | None = None) -> Iterable[Case]:
        """Load MemBench cases."""
        if self.agent_type not in ("FirstAgent", "ThirdAgent"):
            raise ValueError(
                f"Invalid agent_type '{self.agent_type}'. Available: ['FirstAgent', 'ThirdAgent']"
            )

        cache_dir = Path.home() / ".cache" / "agentbench"
        repo_dir = cache_dir / "Membench"
        _ensure_membench_repo(repo_dir)

        source_dir = repo_dir / "MemData" / self.agent_type
        if not source_dir.exists():
            raise FileNotFoundError(f"MemBench data not found at {source_dir}")

        records_with_names: list[tuple[dict, str]] = []
        for json_path in sorted(source_dir.glob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                records = data.get("events") or data.get("multi_agent") or []
                for record in records:
                    records_with_names.append((record, json_path.stem))
                    if limit is not None and len(records_with_names) >= limit:
                        break

                if limit is not None and len(records_with_names) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Failed to load MemBench file {json_path}: {e}")
                continue

        records_with_names.sort(key=lambda r: r[0]["tid"])

        if limit is not None:
            records_with_names = records_with_names[:limit]

        return [
            _record_to_case(record, dataset_name) for record, dataset_name in records_with_names
        ]

    async def eval(
        self,
        case: Case,
        output: str | None,
        error: str | None,
        judge_model: str = "gpt-4o-mini",
        judge_client: AsyncOpenAI | None = None,
    ) -> tuple[EvaluationResult, dict[str, int] | None]:
        """Evaluate MemBench case using F1 and/or LLM judge."""
        if error or not output:
            return EvaluationResult(passed=False), None

        if "choices" in case.inputs:
            passed = self._evaluate_multiple_choice(case, output)
            return EvaluationResult(passed=passed, f1_passed=passed), None
        else:
            f1_score, f1_passed = self._evaluate_f1_score(case, output)
            judge_passed, judge_metrics = await self._evaluate_llm_judge(
                case, output, judge_model, judge_client
            )
            return (
                EvaluationResult(
                    passed=f1_passed and judge_passed,
                    f1_score=f1_score,
                    f1_passed=f1_passed,
                    judge_passed=judge_passed,
                ),
                judge_metrics,
            )

    def _evaluate_multiple_choice(self, case: Case, output: str) -> bool:
        """Evaluate multiple choice answer by comparing first letter."""
        expected = case.expected_output
        if not output or not expected:
            return False

        response = output.strip().upper()
        choice = response[0] if response else ""
        expected_choice = expected.strip().upper()
        return choice == expected_choice

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
