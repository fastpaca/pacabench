"""LongMemEval dataset loader."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from agentbench.stages.case import Case


def load_longmemeval(
    split: str = "s_cleaned",
    limit: int | None = None,
    cache_dir: Path | str | None = None,
) -> list[Case]:
    """
    Load LongMemEval dataset.

    Args:
        split: "s_cleaned" (single-session) or "m_cleaned" (multi-session)
        limit: Optional limit on number of samples
        cache_dir: Optional cache directory for HuggingFace downloads

    Returns:
        List of Cases
    """
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
        cache_dir=str(cache_dir) if cache_dir else None,
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


def _flatten_sessions(sessions: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    """Flatten nested session structure into a single conversation history."""
    turns = []

    for session in sessions:
        if isinstance(session, list):
            turns.extend(session)
        elif isinstance(session, dict):
            turns.append(session)

    return [turn for turn in turns if isinstance(turn, dict) and turn.get("content")]
